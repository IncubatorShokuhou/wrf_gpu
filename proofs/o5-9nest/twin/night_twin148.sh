#!/bin/bash
# #148 CPU twin, night window (ALISIOS GO cross TO_WRF_GPU2_ADMISSIONS_DONE_AND_CPU_TWIN_20261024.md; start ONLY on the manager's
# GO after Nightly-DONE). Original CPU-WRF 4.7.1 (1d0f5779), n18k5 (d01,d02,d04,d05,d06), nproc 3x2 = 6 ranks, one per logical
# CPU on the granted cores 5,6,7 + SMT 21,22,23, nice 19, E219 CPU-only OpenMPI env, memcg via with_ram_slot (<= 24 GiB grant),
# hard deadline 06:00Z, registry real. Stop (ALISIOS return within 5 min): proofs/o5-9nest/twin/stop_twin.sh ROOT.
# Pre-flight refuses (rc 3) on QUIET, <DATA_ROOT> < 365 + 6 GiB free, an existing ROOT/cpu/wrfout, or any foreign process with
# > 20 % CPU on the granted cores (ps psr sample).
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/o5-9nest/twin_k5
CPUS=5,6,7,21,22,23
DEADLINE=${DEADLINE:-2026-10-25T06:00:00Z}
RAM=${RAM:-12G}
LOG=<USER_HOME>/wrf_gpu2_lanes/o5-9nest/night_twin.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
[ -e /tmp/wrf_gpu2_quiet ] && { log "QUIET -> refuse"; exit 3; }
FREE=$(df -BG --output=avail <DATA_ROOT> | tail -1 | tr -dc 0-9)
(( FREE >= 371 )) || { log "<DATA_ROOT> free ${FREE}G < 371G -> refuse"; exit 3; }
[ -s "$ROOT/PREPARE.json" ] || { log "not prepared: $ROOT"; exit 3; }
ls "$ROOT"/cpu/wrfout_* >/dev/null 2>&1 && { log "outputs exist -> refuse (fresh root needed)"; exit 3; }
BUSY=$(ps -eo pid=,psr=,pcpu=,comm= | awk -v c=",$CPUS," 'index(c, ","$2",") && $3 > 20 {print}')
[ -n "$BUSY" ] && { log "granted cores busy -> refuse: $BUSY"; exit 3; }
log "PREFLIGHT OK free=${FREE}G cpus=$CPUS deadline=$DEADLINE ram=$RAM launcher_sha=$(sha256sum "$0" | cut -c1-16) runner_sha=$(sha256sum "$HERE/run_twin148.py" | cut -c1-16)"
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { log "PREFLIGHT_ONLY: nothing started"; exit 0; }
cd "$HERE/../../.." || exit 3
JAX_PLATFORMS=cpu scripts/with_ram_slot.sh --label o5-9nest-twin148 --max "$RAM" --timeout 1800 -- \
  python3 "$HERE/run_twin148.py" run --root "$ROOT" --cpus "$CPUS" --deadline-utc "$DEADLINE" --registry real --need-gib 6 >> "$LOG" 2>&1
RC=$?
log "END rc=$RC"
exit $RC
