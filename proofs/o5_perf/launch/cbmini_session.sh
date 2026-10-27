#!/usr/bin/env bash
# o5-perf CUDA MINI for cbab (manager 17:03Z): ONE consumer view, two warm PROD 3 h S0 arms (sealed OFF, then patched V1p =
# min graph size 1 on the PROD d01/d02 step executables), compile-free. Checks: both arms' 6 wrfout SHA == sealed bundle receipt
# B6_prod_3h (S0 harness == CLI output, HERO W0 proof); d02 host ms/call and seg2-3 ms/root OFF vs V1p; VLOG: d02 rejected for
# min graph size under OFF, command buffer created under V1p. Expected lock ~2.5 min (GPU busy ~80 s), cap 5 min (ALISIOS mini limit), 120 s per arm.
set -uo pipefail
umask 022
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf
TAG=${O5_RUN_TAG:?run tag}
R=$S/cbmini/$TAG
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
SMOKE=${O5_LAUNCH_SMOKE:-0}
LAYER=$S/layer_v043_cbmini; [[ "$SMOKE" == 1 ]] && LAYER=${LAYER}_smoke
FLOOR=365; [[ "$SMOKE" == 1 ]] && FLOOR=200; BUD=120; [[ "$SMOKE" == 1 ]] && BUD=45
mkdir -p "$R"
[[ "$(git -C "$P/src" rev-parse --short=10 HEAD)" == 888804310b ]] || { echo PIN_MISMATCH; exit 4; }
[[ -s "$S/cbmini_patched/PATCH_RECEIPT.json" ]] || { echo NO_PATCHED_SET; exit 4; }
free=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' '); (( free >= FLOOR )) || { echo "DISK_FLOOR ${free}"; exit 95; }
[[ -e "$LAYER" ]] && { echo "LAYER_EXISTS $LAYER (fresh layer required)"; exit 4; }
bash "$P/tools/bundle_consumer.sh" --no-verify --bundle "$P" --layer "$LAYER" -- bash "$L/cbmini_inner.sh" "$R" "$BUD" \
  > "$R/consumer.log" 2>&1
echo "consumer rc=$? $(tail -1 "$R/consumer.log")"
python3 -I "$L/cbmini_summary.py" "$R" "$P/receipts/B6_prod_3h/wrfout_sha.txt"
echo "cbmini session done $TAG"
