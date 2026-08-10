#!/usr/bin/env bash
# run_one.sh "<jobid>\t<donefile>\t<command>"
#
# Skips completed jobs, retries once, never propagates a failure to the queue.
#
# Admission control. A ToN-IoT raw-byte job holds about 20 GB resident while it
# builds tensors for 8.36 M flows; five of them at once exhausted 125 GB and the
# kernel OOM-killer took out this queue and other people's containers with it.
# A job now waits until enough memory is actually free, and holds a reservation
# over its ramp-up so that concurrent jobs do not all read the same free memory
# and start together. Once the allocation is visible in MemAvailable the
# reservation is dropped and the real number speaks for itself.
RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$RUNS/.." && pwd)"
RESV="$RUNS/.reserved_gb"
LOCK="$RUNS/.admit.lock"
# Two floors, because one floor starves the big jobs. A ToN-IoT job needs
# 22 GB and the light queues take memory in 9-10 GB bites, so with a single
# floor the large allocation never finds a window: two of them waited 76 and 56
# minutes while small jobs walked past them. Light jobs must now leave enough
# room for a ToN-IoT job to land, which is what reserves the space.
# Floors sized by arithmetic rather than by feel, which is what the earlier
# values were. A ToN-IoT job settles at 21 GB, a light one at about 7. Admitting
# a job at the floor must still leave a usable margin, so the floor is what the
# job takes plus roughly 24 GB of headroom for the rest of the machine.
# 40 rather than 45, so that two ToN-IoT trainers can overlap when the memory is
# there: with 67 GB free the first takes 24 and the second still sees 43. A third
# cannot follow, because 19 GB is below the floor, so the arithmetic caps the
# overlap at two without needing a separate counter. ToN-IoT is the critical path
# with 23 jobs left, and one at a time makes it five hours instead of three.
FLOOR_BIG=40        # 21 GB job + 19 GB margin, overlap capped at two
# Reserving 46 GB permanently for ToN-IoT was wrong: only one such job runs at a
# time and it holds 21 GB, so the rest sat idle while the random forest, which
# needs 46 to start, was blocked at 42 available. A ToN-IoT job does not need the
# room reserved continuously, only at the moment one finishes and releases its
# 21 GB in a block, which lifts the machine over FLOOR_BIG on its own.
FLOOR_SMALL=26      # keep 26 GB clear; the door for a big job opens on release

line="$1"
jid="${line%%$'\t'*}"
rest="${line#*$'\t'}"
done_file="${rest%%$'\t'*}"
cmd="${rest#*$'\t'}"

if [ -f "$done_file" ]; then
    echo "$(date +%H:%M:%S) SKIP  $jid"
    exit 0
fi

# Memory requirement and ramp-up window, by job kind and testbed.
FLOOR=$FLOOR_SMALL
case "$jid" in
    byte:toniot:*)       NEED=24; WARM=420; FLOOR=$FLOOR_BIG; CAP=5400 ;;
    # The floor belongs to the job, not to the testbed. Giving every ToN-IoT job
    # the 45 GB floor of the byte trainer stalled the ToN-IoT random forest, which
    # holds 14 GB, for an hour and three quarters: it waited on three times the
    # room it needed. Each floor is now the job's own footprint plus about 16 GB.
    occl:toniot:*)       NEED=18; WARM=150; FLOOR=34; CAP=3600 ;;
    rf:toniot:*)         NEED=14; WARM=150; FLOOR=30; CAP=5400 ;;
    byte:cic_ids2018:*)  NEED=10; WARM=120; CAP=3600 ;;
    byte:cic_iot2023:*)  NEED=9;  WARM=120; CAP=3600 ;;
    byte:cic_ids2017:*)  NEED=10; WARM=120; CAP=3600 ;;
    occl:*)              NEED=9;  WARM=100; CAP=2400 ;;
    rf:*)                NEED=10; WARM=100; CAP=3600 ;;
    *)                   NEED=12; WARM=150; CAP=3600 ;;
esac

avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

# A 24 GB request cannot win against a stream of 7 GB requests in a first-come
# scheme: shifting floors only moved the starvation between job kinds, and a
# ToN-IoT trainer ended up waiting four and a half hours. So a big job that has
# waited raises a flag, and small jobs then stop consuming the memory it needs
# until it gets in. This is a reservation, which is what the situation calls for,
# rather than another threshold.
BIGFLAG="$RUNS/.big_waiting"
BIG=0
case "$jid" in byte:toniot:*) BIG=1 ;; esac

reserve() {
    local waited=0
    while true; do
        exec 9>"$LOCK"; flock 9
        local avail held budget
        avail=$(avail_gb)
        held=$(cat "$RESV" 2>/dev/null || echo 0)
        # Ageing. The ledger double counts a job between its start and the
        # moment its allocation shows up in MemAvailable, and a stream of small
        # jobs keeps it permanently non-zero. A large request then never finds a
        # window: ToN-IoT jobs waited over an hour twice this way. After five
        # minutes of waiting the request stops subtracting the ledger and trusts
        # MemAvailable alone, which the floor already makes safe.
        budget=$((avail - held))
        local floor=$FLOOR

        if [ "$BIG" -eq 1 ]; then
            # Raise the flag only when ToN-IoT is actually starved, not merely
            # queued. With 22 of them waiting there is always one in line, so a
            # flag raised on queueing alone stayed up permanently and the random
            # forest stood aside for forty minutes with nothing running.
            local bigrun
            bigrun=$(pgrep -u "$USER" -f "toniot/train_byte" 2>/dev/null | while read -r q; do
                       [ "$(cat /proc/$q/comm 2>/dev/null)" = python ] && echo x; done | wc -l)
            if [ "$waited" -ge 240 ] && [ "$bigrun" -lt 2 ]; then
                date +%s > "$BIGFLAG"
            fi
        elif [ -f "$BIGFLAG" ]; then
            # Stand off for the big job, but only far enough to leave it room,
            # and only briefly, so the flag can never wedge the queue.
            local age=$(( $(date +%s) - $(cat "$BIGFLAG" 2>/dev/null || echo 0) ))
            # Courtesy has a limit. With eighteen ToN-IoT jobs still queued there
            # is nearly always one waiting, so the flag kept being re-raised and
            # the random forest stood aside indefinitely: two of its jobs waited
            # 59 and 37 minutes with nothing of theirs running. A job that has
            # itself waited ten minutes stops yielding.
            if [ "$age" -lt 600 ] && [ "$waited" -lt 600 ]; then
                floor=$((FLOOR_BIG + NEED))
            elif [ "$age" -ge 600 ]; then
                rm -f "$BIGFLAG"
            fi
        fi
        # Ageing has to buy its way past the ledger. Ignoring the ledger outright
        # let two ToN-IoT jobs pass in the same window, before either had
        # allocated, and the machine went to 105 GB of 125. A job that bypasses
        # the ledger must instead see room for itself on top of the floor, which
        # only happens when the memory is genuinely there.
        # The ageing bypass is now exclusive. Ignoring the ledger is safe for one
        # job and catastrophic for eight: every waiting job crosses the five
        # minute mark at about the same time, so they all stopped reading the
        # ledger and all started together. Eight of them landed within fifteen
        # seconds, the ledger claimed 139 GB of 125, and the machine stopped
        # answering ssh. Only the holder of this token may bypass.
        if [ "$waited" -ge 300 ] && exec 7>"$RUNS/.bypass.lock" && flock -n 7; then
            budget=$avail
            # Take the higher of the two floors. Assigning here instead would
            # undo the stand-off above the moment a small job had waited five
            # minutes, which is exactly when the big job needs it to hold.
            [ $((FLOOR + NEED)) -gt "$floor" ] && floor=$((FLOOR + NEED))
        fi
        if [ "$avail" -ge "$floor" ] && [ "$budget" -ge "$NEED" ]; then
            echo $((held + NEED)) > "$RESV"
            flock -u 7 2>/dev/null; exec 7>&- 2>/dev/null
            [ "$BIG" -eq 1 ] && rm -f "$BIGFLAG"
            flock -u 9; exec 9>&-
            [ "$waited" -gt 0 ] && echo "$(date +%H:%M:%S) ADMIT $jid after ${waited}s"
            return 0
        fi
        flock -u 9; exec 9>&-
        flock -u 7 2>/dev/null; exec 7>&- 2>/dev/null
        sleep $((20 + RANDOM % 25)); waited=$((waited + 30))
    done
}

release() {
    exec 9>"$LOCK"; flock 9
    local held n
    held=$(cat "$RESV" 2>/dev/null || echo 0)
    n=$((held - NEED)); [ "$n" -lt 0 ] && n=0
    echo "$n" > "$RESV"
    flock -u 9; exec 9>&-
}

# An occlusion sweep reads a model the byte queue has not necessarily trained
# yet. Defer rather than fail: a missing prerequisite is a matter of ordering,
# and recording it as a failure both burns a slot and hides the real failures.
model=$(echo "$cmd" | grep -oE -- "--model-path [^ ]+" | cut -d' ' -f2)
if [ -n "$model" ] && [ ! -f "$model" ]; then
    echo "$(date +%H:%M:%S) DEFER $jid  (model not trained yet)"
    exit 0
fi

slug=$(echo "$jid" | tr ':/' '__')
log="$RUNS/logs/$slug.log"

# Non-blocking claim, so a second queue on the same manifest skips this job
# rather than running it into the same output directory.
mkdir -p "$RUNS/claims"
exec 8>"$RUNS/claims/$slug.lock"
if ! flock -n 8; then
    echo "$(date +%H:%M:%S) BUSY  $jid"
    exit 0
fi

start=$SECONDS
for attempt in 1 2; do
    reserve
    # If this queue does overrun memory again, the kernel must pick one of our
    # jobs and not somebody else's service. choom raises the score for the
    # child only; writing /proc/self/oom_score_adj from the subshell did not
    # survive into the python process.
    ( cd "$ROOT" \
      && exec timeout -k 30 "$CAP" choom -n 800 -- bash -c "$cmd" ) > "$log" 2>&1 &
    job=$!
    sleep "$WARM"          # let the allocation become visible, then stop double counting
    release
    wait "$job"
    if [ -f "$done_file" ]; then
        echo "$(date +%H:%M:%S) OK    $jid  $((SECONDS-start))s  try$attempt"
        exit 0
    fi
    if [ $((SECONDS-start)) -ge "$CAP" ]; then
        echo "$(date +%H:%M:%S) HUNG  $jid  (killed at ${CAP}s, attempt $attempt)"
    fi
    if grep -qiE "out of memory|CUDA out of memory|Killed" "$log" 2>/dev/null; then
        echo "$(date +%H:%M:%S) OOM   $jid  (attempt $attempt)"
        sleep 120
    fi
    [ $attempt -eq 1 ] && echo "$(date +%H:%M:%S) RETRY $jid"
done
echo "$(date +%H:%M:%S) FAIL  $jid  (see $log)"
printf '%s\t%s\n' "$jid" "$log" >> "$RUNS/failed.txt"
exit 0
