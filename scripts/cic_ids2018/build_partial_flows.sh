#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Skip Tuesday-20-02-2018 (needs 100+ GB, run separately when RF is done)
DAYS="Wednesday-14-02-2018 Thursday-15-02-2018 Friday-16-02-2018 Friday-02-03-2018"
RATIOS="0.75 0.50 0.25 0.10"

for ratio in $RATIOS; do
  pct_int=$(echo "$ratio * 100" | bc | cut -d. -f1)
  pct_tag=$(printf "pct_%03d" "$pct_int")
  for day in $DAYS; do
    echo "--- $pct_tag / $day ---"
    conda run -n earlynids python3 -c "
from src.cic_ids2018.pcap_extracted_flow_pipeline import build_day_partial_flow
import time
t0 = time.time()
m = build_day_partial_flow('$day', observation_ratio=$ratio, force=False)
elapsed = time.time() - t0
n = m.get('stats', {}).get('${pct_tag}_flow_rows', '?')
print(f'DONE $pct_tag $day: {elapsed:.1f}s, {n} flows')
" 2>&1
    echo ""
  done
done
echo "ALL DONE (excluding Tuesday-20-02-2018)"

# =============================================================================
# Time-based partial-flow grid (Phase A)
# - time_abs: truncate each flow at first N seconds (absolute window)
# These live alongside pct_XXX and never overwrite them.
#
# NOTE: time_pct (first X% of a flow's own duration) was removed. It is an oracle
# cut — the threshold needs the flow's total duration, unknown online — so it
# overstates early-detection performance. Use deployable cuts only (packet-count
# pct_XXX, packet_abs_N, time_abs_Ns).
# =============================================================================

TIME_DAYS="Wednesday-14-02-2018 Thursday-15-02-2018 Friday-16-02-2018 Friday-23-02-2018 Thursday-22-02-2018 Friday-02-03-2018 Tuesday-20-02-2018"

# (cut_mode, cut_value, tag_label) triples.
TIME_SPECS=(
  "time_abs 1.0  time_abs_1s"
  "time_abs 3.0  time_abs_3s"
)

for spec in "${TIME_SPECS[@]}"; do
  # shellcheck disable=SC2086
  set -- $spec
  cut_mode="$1"
  cut_value="$2"
  tag="$3"
  for day in $TIME_DAYS; do
    echo "--- $tag / $day ---"
    conda run -n earlynids python3 -c "
from src.cic_ids2018.pcap_extracted_flow_pipeline import build_day_partial_flow
import time
t0 = time.time()
m = build_day_partial_flow('$day', cut_mode='$cut_mode', cut_value=$cut_value, force=False)
elapsed = time.time() - t0
n = m.get('stats', {}).get('${tag}_flow_rows', '?')
print(f'DONE $tag $day: {elapsed:.1f}s, {n} flows')
" 2>&1
    echo ""
  done
done
echo "ALL TIME-BASED DONE"
