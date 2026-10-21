#!/usr/bin/env bash
# #148 fused-sibling independence, GPU phase 2 (IMMUTABLE once its sha is posted).
# Window 2026-10-21 07:32-07:57Z (alisios_mgr grant), lease /tmp/wrf_gpu2_run_approvals/w43-148fid.lease.
# Lock wait ends by 07:42Z. The deadline is computed INSIDE the locked command: REM = 07:55Z - now - 40 s
# (refuse < 420 s); copies = 3 if REM >= 1200 s else 2. The measured python runs under that inner timeout inside
# the lock, so no GPU process outlives the lock and everything ends by 07:55Z. Release defaults (product kernels),
# real PROD d01 + N copies of d02 (B perturbs the middle sibling, C the first), ORD+RAD phases, --require-gpu,
# AOT off, private autotune dir; mutant + isolated arm were validated in CPU phase 1 (same script file).
# CLOCK SKEW (manager 2026-10-20): every window time is REAL UTC from real_utc.sh (system epoch + chrony offset), never
# raw date. While chrony slews, system-clock DURATIONS (lock --timeout, coreutils timeout) can run up to ~9 % long
# (maxslewrate 83333 ppm), so both are scaled by 0.9; every log/MAILBOX line records real and system time.
set -u
W=<DATA_ROOT>/wrf_gpu2/worktrees/w43-148fid
M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
clocks() { echo "real $($RU --iso) sys $(date -u +%FT%TZ)"; }
START=$(date -u -d "2026-10-21T07:32:00Z" +%s)
LOCK_BY=$(date -u -d "2026-10-21T07:42:00Z" +%s)
END=$(date -u -d "2026-10-21T07:55:00Z" +%s)
cd "$W" || exit 7
now=$($RU)
case "$now" in ''|*[!0-9]*) echo "real_utc.sh failed"; exit 7;; esac
if [ "$now" -lt "$START" ]; then echo "too early ($(clocks))"; exit 4; fi
[ -e /tmp/wrf_gpu2_run_approvals/w43-148fid.lease ] || { echo "no lease"; exit 5; }
WAIT=$(( LOCK_BY - now ))
if [ "$WAIT" -gt 300 ]; then WAIT=300; fi
if [ "$WAIT" -lt 10 ]; then echo "lock deadline 07:42Z (real) passed ($(clocks))"; exit 6; fi
WAIT=$(( WAIT * 9 / 10 ))
SHA=$(sha256sum analysis/sibling/sibling_test.py | cut -c1-12)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-148fid: GPU START [$(clocks)] sibling phase 2 (release defaults, PROD d01 + 3 or 2 x d02 chosen inside the lock, ORD+RAD, sibling_test.py sha $SHA, lock wait <= ${WAIT}s, hard end 07:55Z real)" >> "$M"
export W END SHA RU
"$LOCK" --label w43-148fid --timeout "$WAIT" -- bash -c '
  set -u
  now=$($RU)
  case "$now" in ""|*[!0-9]*) echo "inside lock: real_utc.sh failed"; exit 7;; esac
  REM=$(( END - now - 40 ))
  if [ "$REM" -lt 420 ]; then echo "inside lock: window too short (${REM}s real)"; exit 6; fi
  if [ "$REM" -ge 1200 ]; then COPIES=3; else COPIES=2; fi
  TSYS=$(( REM * 9 / 10 ))
  OUT="$W/analysis/data/sibling_gpu2"; SMI="$W/analysis/data/sibling_gpu2_nvidia_smi_start.txt"
  { echo "real $($RU --iso) sys $(date -u +%FT%TZ)"; echo "REM=$REM real s, inner timeout TSYS=$TSYS sys s, COPIES=$COPIES"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv;
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$SMI" 2>&1
  echo "inside lock: REM=${REM}s real, timeout ${TSYS}s sys, copies=$COPIES"
  env -u JAX_PLATFORMS GPUWRF_JAX_CACHE=0 GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$W/analysis/data/gpu_autotune" \
      PYTHONPATH="$W/src" SIBLING_SMI_FILE="$SMI" \
      timeout -k 15 "$TSYS" python -u "$W/analysis/sibling/sibling_test.py" --out "$OUT" --copies "$COPIES" \
      --mode release --skip-isolated --skip-mutant --require-gpu > "$W/analysis/data/sibling_gpu2.log" 2>&1
'
rc=$?
OUT=$W/analysis/data/sibling_gpu2
V=$(python3 -c "import json;d=json.load(open('$OUT/log.json'));print(d.get('verdict','NONE'), 'copies', d['args'].get('copies'), 'platform', d.get('platform'))" 2>/dev/null || echo NONE)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w43-148fid: GPU END [$(clocks)] rc=$rc verdict=$V (log $W/analysis/data/sibling_gpu2.log)" >> "$M"
exit $rc
