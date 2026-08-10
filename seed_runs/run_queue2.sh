#!/usr/bin/env bash
# run_queue2.sh <manifest> <slots> [cpu-affinity]
# Resumable: a job whose done-file exists is skipped, so re-running the queue
# after any interruption picks up exactly where it stopped. Slots are an upper
# bound only; run_job.sh admits a job when the memory it needs is actually free.
set -u
RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$1"
SLOTS="${2:-4}"
AFFINITY="${3:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

total=$(wc -l < "$MANIFEST")
echo "=== queue $(basename "$MANIFEST"): $total jobs, $SLOTS slots, OMP=$OMP_NUM_THREADS ${AFFINITY:+affinity=$AFFINITY} ==="
date

runner="bash $RUNS/run_job.sh"
[ -n "$AFFINITY" ] && runner="taskset -c $AFFINITY bash $RUNS/run_job.sh"

xargs -d '\n' -P "$SLOTS" -I LINE $runner "LINE" < "$MANIFEST"

echo "=== queue $(basename "$MANIFEST") drained ==="
date
