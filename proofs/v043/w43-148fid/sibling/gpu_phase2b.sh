#!/usr/bin/env bash
# #148 fused-sibling independence, GPU phase 2 RUN 2 (IMMUTABLE once its sha is posted; new file per E163, run 1 =
# gpu_phase2.sh). Run 1 (2026-10-21 07:32Z) timed out: every perturbed arm recompiled fused_jit because the v1
# perturbation changed leaf jit signatures (see REPORT). sibling_test.py now perturbs values only, asserts identical
# leaf signatures before any arm, counts fused_jit traces/compiles per stamp and, with --abort-on-recompile, stops
# rc 3 at the first second compile. Candidate window 2026-10-21 15:20-16:20Z REAL UTC (manager), only after the
# manager lease line. Lock wait ends by 15:50Z; inside the lock REM = 16:15Z - now - 40 s (refuse < 900 s); copies 3
# if REM >= 1200 s else 2. The python runs under that inner timeout inside the lock, so nothing outlives the lock.
# Release defaults, real PROD d01 + N x d02 (B perturbs the middle sibling, C the first), ORD+RAD, --require-gpu,
# AOT off, private autotune dir; mutant + isolated arm were validated in CPU phase 1. Times via real_utc.sh; both
# clocks logged; system-clock durations scaled 0.9 (chrony slew guard, kept from run 1).
set -u
W=<DATA_ROOT>/wrf_gpu2/worktrees/w43-148fid
M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
clocks() { echo "real $($RU --iso) sys $(date -u +%FT%TZ)"; }
START=$(date -u -d "2026-10-21T15:20:00Z" +%s)
LOCK_BY=$(date -u -d "2026-10-21T15:50:00Z" +%s)
END=$(date -u -d "2026-10-21T16:15:00Z" +%s)
cd "$W" || exit 7
now=$($RU)
case "$now" in ''|*[!0-9]*) echo "real_utc.sh failed"; exit 7;; esac
if [ "$now" -lt "$START" ]; then echo "too early ($(clocks))"; exit 4; fi
[ -e /tmp/wrf_gpu2_run_approvals/w43-148fid.lease ] || { echo "no lease"; exit 5; }
WAIT=$(( LOCK_BY - now ))
if [ "$WAIT" -gt 1800 ]; then WAIT=1800; fi
if [ "$WAIT" -lt 10 ]; then echo "lock deadline 15:50Z (real) passed ($(clocks))"; exit 6; fi
WAIT=$(( WAIT * 9 / 10 ))
SHA=$(sha256sum analysis/sibling/sibling_test.py | cut -c1-12)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-148fid: GPU START [$(clocks)] sibling phase 2 run 2 (release defaults, PROD d01 + 3 or 2 x d02 chosen inside the lock, ORD+RAD, one fused compile, sibling_test.py sha $SHA, lock wait <= ${WAIT}s, hard end 16:15Z real)" >> "$M"
export W END SHA RU
"$LOCK" --label w43-148fid --timeout "$WAIT" -- bash -c '
  set -u
  now=$($RU)
  case "$now" in ""|*[!0-9]*) echo "inside lock: real_utc.sh failed"; exit 7;; esac
  REM=$(( END - now - 40 ))
  if [ "$REM" -lt 900 ]; then echo "inside lock: window too short (${REM}s real)"; exit 6; fi
  if [ "$REM" -ge 1200 ]; then COPIES=3; else COPIES=2; fi
  TSYS=$(( REM * 9 / 10 ))
  OUT="$W/analysis/data/sibling_gpu3"; SMI="$W/analysis/data/sibling_gpu3_nvidia_smi_start.txt"
  { echo "real $($RU --iso) sys $(date -u +%FT%TZ)"; echo "REM=$REM real s, inner timeout TSYS=$TSYS sys s, COPIES=$COPIES"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv;
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$SMI" 2>&1
  echo "inside lock: REM=${REM}s real, timeout ${TSYS}s sys, copies=$COPIES"
  env -u JAX_PLATFORMS GPUWRF_JAX_CACHE=0 GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$W/analysis/data/gpu_autotune" \
      PYTHONPATH="$W/src" SIBLING_SMI_FILE="$SMI" \
      timeout -k 15 "$TSYS" python -u "$W/analysis/sibling/sibling_test.py" --out "$OUT" --copies "$COPIES" \
      --mode release --skip-isolated --skip-mutant --require-gpu --abort-on-recompile > "$W/analysis/data/sibling_gpu3.log" 2>&1
'
rc=$?
OUT=$W/analysis/data/sibling_gpu3
V=$(python3 -c "import json;d=json.load(open('$OUT/log.json'));c=d.get('compile_counts',{});print(d.get('verdict','NONE'), 'copies', d['args'].get('copies'), 'platform', d.get('platform'), 'fused_compiles', (list(c.values())[-1] if c else {}).get('fused_compile'))" 2>/dev/null || echo NONE)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-148fid: GPU END [$(clocks)] rc=$rc verdict=$V (log $W/analysis/data/sibling_gpu3.log)" >> "$M"
exit $rc
