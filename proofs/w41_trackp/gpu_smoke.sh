#!/usr/bin/env bash
# Track P GPU smoke, ONE arm per call (= one GPU lock session; the caller wraps it in scripts/with_gpu_lock.sh;
# the GPU process runs in its own 20G systemd scope, never inside a RAM slot).
# usage: gpu_smoke.sh <src_root> <work_dir> <arm>      arm: S0 S1 S2 P0 P1 P2
#   work_dir holds controls/{swiss_prior,swiss_T,prod_prior,prod_A,prod_B}.npz and case/swiss (Swiss inputs, no wrfout)
# S0 Swiss 24 h OFF (cold) · S1 Swiss 24 h PRIOR (cold, controls program) · S2 Swiss 24 h T (warm: must NOT compile)
# P0 PROD 3 h PRIOR (cold) · P1/P2 PROD 3 h A/B (warm: must NOT compile). Release defaults, private caches.
# The device is asserted in the measured process AFTER the run (asserting first would initialise the backend
# before the CLI's C-auto pool configuration, E190). Cache file counts before/after expose any recompile.
set -u
SRC=$(cd "$1" && pwd); W=$(cd "$2" && pwd); ARM=$3
dom=""   # PROD arms run both nests (the control files carry d01 + d02; the CLI refuses entries outside the run)
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
case $ARM in
  S0) name=S0_swiss_off;   cache=$W/cache_swiss; input=$W/case/swiss; hours=24; ctl=;                         lim=1500 ;;
  S1) name=S1_swiss_prior; cache=$W/cache_swiss; input=$W/case/swiss; hours=24; ctl=$W/controls/swiss_prior.npz; lim=1500 ;;
  S2) name=S2_swiss_T;     cache=$W/cache_swiss; input=$W/case/swiss; hours=24; ctl=$W/controls/swiss_T.npz;     lim=900 ;;
  P0) name=P0_prod_prior;  cache=$W/cache_prod;  input=$PROD;         hours=3;  ctl=$W/controls/prod_prior.npz;  lim=1800; dom="--max-dom 2" ;;
  P1) name=P1_prod_A;      cache=$W/cache_prod;  input=$PROD;         hours=3;  ctl=$W/controls/prod_A.npz;      lim=1800; dom="--max-dom 2" ;;
  P2) name=P2_prod_B;      cache=$W/cache_prod;  input=$PROD;         hours=3;  ctl=$W/controls/prod_B.npz;      lim=1800; dom="--max-dom 2" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
[ -e /tmp/wrf_gpu2_quiet ] && { echo QUIET; exit 3; }
mkdir -p "$cache" "$W/runs"
out=$W/runs/$name; rm -rf "$out"
n0=$(find "$cache" -type f | wc -l)
t0=$(date -u +%s.%N)
# own host-memory cap inside the lock (manager rule 11:37Z: never with_ram_slot around/inside the GPU lock)
( cd "$SRC" && systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 -- \
    env PYTHONPATH="$SRC/src" GPUWRF_JAX_CACHE_DIR="$cache" JAX_COMPILATION_CACHE_DIR="$cache" \
    timeout "$lim" python -c 'import sys
from gpuwrf.cli import main
rc = main(sys.argv[1:])
import jax
d = jax.devices()[0]
print("ASSERT_DEVICE", d.platform, d.device_kind, flush=True)
sys.exit(rc if d.platform == "gpu" else 5)' run --input-dir "$input" --output-dir "$out" --hours "$hours" $dom \
    ${ctl:+--surface-controls "$ctl"} ) > "$W/runs/$name.log" 2>&1
rc=$?
t1=$(date -u +%s.%N); n1=$(find "$cache" -type f | wc -l)
dev=$(grep -m1 ASSERT_DEVICE "$W/runs/$name.log" | cut -d' ' -f2- )
python3 - "$name" "$rc" "$t0" "$t1" "$n0" "$n1" "$ctl" "$dev" <<'EOF' | tee -a "$W/arms.jsonl"
import json, sys
name, rc, t0, t1, n0, n1, ctl, dev = sys.argv[1:]
print(json.dumps({"arm": name, "rc": int(rc), "wall_s": round(float(t1) - float(t0), 1), "cache_files_before": int(n0),
                  "cache_files_after": int(n1), "controls": ctl, "device": dev}))
EOF
exit $rc
