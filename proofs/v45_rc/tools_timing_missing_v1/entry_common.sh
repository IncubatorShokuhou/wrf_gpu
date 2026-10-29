# v45-rc v0.4.5 GPU entry body (sourced by the four entry files; never run directly). The entry files differ ONLY in
# their header constants: ENTRY_PATH ENTRY_NAME ROLE SESSION TABLE CAP_MIN START_FORECAST_GIB POST_RESERVE_S POST
# TOOLS_SUMS_SHA (see each file).
#
# Lifecycle (manager sequencer night_seq_v9.sh runs `bash <entry>` in its own process group with GA_START_UTC/GA_END_UTC):
#  outer   tools SHA256SUMS == pinned sha; QUIET; GA window (real UTC) must hold the whole cap; PIN manifest validated
#          (manifest_check.py); <DATA_ROOT> free - forecast >= 365 GiB; lane lease + with_gpu_lock digest == main pin (GPU);
#          then exec the reviewed one-setsid supervisor (arm_supervisor.py, copy of w44-precompile day_tools_v5) over
#          with_ram_slot.sh (MemoryMax 20G GPU / 12G CPU smoke) over `bash <entry> --owned`.
#  owned   (inside the supervisor's session + RAM-slot cgroup) with_gpu_lock.sh --timeout 120 --label v45-rc ->
#          session payload; lock released -> CPU post (E41 + checker for the short entries, RW3T accept/verdict for RW3)
#          -> ENTRY_RESULT.json -> no own process may remain on the GPU.
# CPU launch smoke: W45R_LAUNCH_SMOKE=1 + W45R_MANIFEST (standin) + W45R_SMOKE_* (see launch_smoke.sh). The ONLY deltas:
#   no with_gpu_lock (and no lease/digest gate), RAM scope 12G (ticket wait up to 1800 s), caps x W45R_SMOKE_CAP_FACTOR, device JAX_PLATFORMS=cpu + CUDA_VISIBLE_DEVICES='' after the
#   product env is recorded (arm.sh), arm duration -> W45R_SMOKE_DURATION or --dry-run per arm table SMOKE_MODE, no
#   nvidia-smi samplers, private QUIET/MAILBOX files, RW3 watch on the smoke cores with non-aborting thresholds, E41 also
#   on a real GPU cache (read-only), seal refusal tolerated (replays then use W45R_SMOKE_REPLAY_BUNDLE read-only).
set -uo pipefail
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/lib.sh
RAM_MAX=20G
RAM_SLOT_TIMEOUT=300   # GPU: slots are frozen for other lanes during a window; CPU smoke: 1800 (FIFO behind other lanes' tests)
TEARDOWN_S=60
: "${ENTRY_PATH:?}" "${ENTRY_NAME:?}" "${ROLE:?}" "${SESSION:?}" "${TABLE:?}" "${CAP_MIN:?}" "${START_FORECAST_GIB:?}" "${POST_RESERVE_S:?}" "${POST:?}"
MODE_ARG=${1:-outer}
# The manager sequencer is the sole smoke selector; no alternate outer command.
if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 && ${1:-outer} == outer ]]; then
  export W45R_LAUNCH_SMOKE=1 W45R_SMOKE_FORECAST_GIB=1 W45R_SMOKE_CAP_FACTOR=1
  export W45R_MANIFEST=<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_timing_missing_v1/PIN_MANIFEST.json
  export W45R_SMOKE_ROOT=<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_timing_missing_v1
  export W45R_SMOKE_DURATION="--hours 1" W45R_SMOKE_CORES=12,28
  export W45R_SMOKE_REPLAY_BUNDLE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088
  export W45R_SMOKE_E41_CACHE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088/jax
  export W45R_SMOKE_E41_PROOF=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/receipts/prime_wn3_20261026T221220Z/A3_wn3_0227_3h/nested_pipeline_run.json
  export W45R_SMOKE_QUIET_FILE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_timing_missing_v1/quiet_private
  export W45R_SMOKE_MAILBOX=<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_timing_missing_v1/mailbox_private.md
  unset W45R_SMOKE_ONLY_SLOTS   # F3: every actual RW3 CLI/consumer/check path runs
fi
SMOKE=0; [[ ${W45R_LAUNCH_SMOKE:-0} == 1 ]] && SMOKE=1
[[ $(w45_sha "$W45R_TOOLS/SHA256SUMS") == "$TOOLS_SUMS_SHA" ]] || { echo "REFUSED: tools SHA256SUMS changed (frozen $TOOLS_SUMS_SHA)"; exit 9; }

if [[ $MODE_ARG == outer ]]; then
  (cd "$W45R_TOOLS" && sha256sum -c --quiet SHA256SUMS) || { echo "REFUSED: a tool file differs from SHA256SUMS"; exit 9; }
  if w45_quiet; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
  if [[ $SMOKE == 1 ]]; then
    : "${W45R_MANIFEST:?smoke needs W45R_MANIFEST}" "${W45R_SMOKE_ROOT:?smoke needs W45R_SMOKE_ROOT}"
    [[ ${GPUWRF_GPU_LOCK_HELD:-0} != 1 ]] || { echo "REFUSED: CPU smoke inside a GPU lock"; exit 2; }
    [ -e /tmp/wrf_gpu2_gpu_window ] && { echo "REFUSED: GPU window open (no new RAM-slot ticket)"; exit 3; }
    MODE=smoke; MAN=$W45R_MANIFEST; RAM_MAX=12G; RAM_SLOT_TIMEOUT=1800
    # CPU runs the arms 2-3x slower than the GPU ([M] Swiss 1 h CPU 794 s vs GPU 3 h 349 s) and the RAM-slot FIFO wait counts
    # against the cap ([M] 7-9 min): the smoke scales the entry cap, the post reserve and every arm cap/expectation by
    # W45R_SMOKE_CAP_FACTOR (smoke only; GPU mode never reads it)
    CAP_MIN=$(( CAP_MIN * ${W45R_SMOKE_CAP_FACTOR:?} )); POST_RESERVE_S=$(( POST_RESERVE_S * W45R_SMOKE_CAP_FACTOR ))
    RUN=$W45R_SMOKE_ROOT/runs/${ENTRY_NAME}_$(date -u +%Y%m%dT%H%M%SZ)_$$
    REC=$W45R_LANE/launch_smoke/receipts/${ENTRY_NAME}_$(date -u +%Y%m%dT%H%M%SZ)
  else
    if compgen -e | grep -q '^W45R_'; then echo "REFUSED: W45R_* variables in a GPU launch: $(compgen -e | grep '^W45R_' | tr '\n' ' ')"; exit 2; fi
    MODE=gpu; MAN=$W45R_TOOLS/PIN_MANIFEST.json
    RUN=$W45R_DATA_SCRATCH/runs/${ENTRY_NAME}_$(date -u +%Y%m%dT%H%M%SZ)_$$
    REC=$W45R_LANE/receipts/${ENTRY_NAME}_$(date -u +%Y%m%dT%H%M%SZ)
  fi
  NOW=$(w45_real_now) || { echo "real_utc.sh failed"; exit 7; }
  START=$(date -u -d "${GA_START_UTC:?GA_START_UTC required}" +%s) || exit 2
  END=$(date -u -d "${GA_END_UTC:?GA_END_UTC required}" +%s) || exit 2
  (( NOW >= START )) || { echo "REFUSED: before the window ($GA_START_UTC)"; exit 4; }
  (( END - NOW >= CAP_MIN * 60 )) || { echo "REFUSED: $(( (END - NOW) / 60 )) min left < cap $CAP_MIN min (GA_END_UTC $GA_END_UTC)"; exit 6; }
  [ -s "$MAN" ] || { echo "REFUSED: no PIN manifest $MAN (run prepare.sh after the pin)"; exit 4; }
  if [[ $MODE == gpu ]]; then
    "$W45R_PY" -I "$W45R_TOOLS/night_admission.py" "$W45R_TOOLS/NIGHTLY_DONE.json" "$GA_START_UTC" "$GA_END_UTC" || exit 4
  fi
  if [[ $MODE == gpu ]]; then "$W45R_PY" -I "$W45R_TOOLS/cpu_lease_check.py" || exit 126; fi
  mkdir -p "$RUN" "$REC" || exit 2
  if [[ $SMOKE == 1 ]]; then touch "$W45R_SMOKE_MAILBOX"; fi
  "$W45R_PY" -I "$W45R_TOOLS/manifest_check.py" "$MAN" "$ROLE" "$MODE" "$RUN/manifest.env" "$RUN/MANIFEST_CHECK.json" || exit 4
  cp "$RUN/MANIFEST_CHECK.json" "$REC/"
  w45_disk_ok "$START_FORECAST_GIB" || { echo "REFUSED: ${W45R_DISK_REFUSAL:-DISK} guard (see measured guard values above)"; exit 95; }
  (( $(w45_free_gib /) >= W45R_ROOT_MIN_GIB )) || { echo "REFUSED: ROOT free < $W45R_ROOT_MIN_GIB GiB"; exit 95; }
  if [[ $MODE == gpu ]]; then
    LEASE=/tmp/wrf_gpu2_run_approvals/v45-rc.lease
    EXP=$(grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
    [[ -n $EXP ]] && (( $(date -u -d "$EXP" +%s 2>/dev/null || echo 0) > $(date -u +%s) + CAP_MIN * 60 )) \
      || { echo "REFUSED: lane lease $LEASE missing/expiring before the cap"; exit 126; }
    [[ $(w45_sha "$W45R_GPU_LOCK") == "$(git -C <USER_HOME>/src/wrf_gpu2 show main:scripts/with_gpu_lock.sha256 | awk '{print $1}')" ]] \
      || { echo "REFUSED: with_gpu_lock.sh digest != main pin"; exit 126; }
  fi
  BUDGET=$(( CAP_MIN * 60 - TEARDOWN_S ))
  (( END - NOW - TEARDOWN_S < BUDGET )) && BUDGET=$(( END - NOW - TEARDOWN_S ))
  export W45R_RUN=$RUN W45R_RECEIPTS=$REC W45R_ENVFILE=$RUN/manifest.env W45R_MODE=$MODE W45R_TABLE=$TABLE \
    W45R_ENTRY_DEADLINE=$(( $(date +%s) + BUDGET )) W45R_ENTRY_NAME=$ENTRY_NAME W45R_ENTRY_PATH=$ENTRY_PATH \
    W45R_MANIFEST_PATH=$MAN
  "$W45R_PY" -I - "$RUN/ENTRY.json" <<PY
import json, os, re, sys
env = dict(device_mode="$MODE", pin_manifest="$MAN")
json.dump(dict(entry="$ENTRY_NAME", path="$ENTRY_PATH", role="$ROLE", mode="$MODE", cap_min=$CAP_MIN, budget_s=$BUDGET,
               start_forecast_gib=$START_FORECAST_GIB, ram_max="$RAM_MAX", window=["$GA_START_UTC", "$GA_END_UTC"],
               real_now=$NOW, manifest="$MAN", tools_sums_sha256="$TOOLS_SUMS_SHA", table="$TABLE", session="$SESSION",
               data_free_gib=$(w45_free_gib <DATA_ROOT>), env=env), open(sys.argv[1], "w"), indent=1)
PY
  cp "$RUN/ENTRY.json" "$REC/"
  echo "ENTRY $ENTRY_NAME mode $MODE run $RUN budget ${BUDGET}s"
  if w45_quiet; then echo QUIET; exit 3; fi
  exec "$W45R_PY" -I "$W45R_TOOLS/arm_supervisor.py" --seconds "$BUDGET" --receipt "$RUN/LIFECYCLE.json" \
    --group-file "$RUN/ARM_GROUP.json" --lock-file /tmp/wrf_gpu2_gpu.lock --term-grace 12 --kill-grace 5 -- \
    systemd-run --user --scope --quiet -p MemoryMax="$RAM_MAX" -p MemorySwapMax=0 -- taskset -c 12,28 nice -n 19 bash "$ENTRY_PATH" --owned
fi

[[ $MODE_ARG == --owned ]] || { echo "usage: bash <entry> (no arguments)"; exit 2; }
: "${W45R_RUN:?}" "${W45R_RECEIPTS:?}" "${W45R_ENVFILE:?}" "${W45R_MODE:?}" "${W45R_ENTRY_DEADLINE:?}"
[[ -n ${W4_SUPERVISOR_PID:-} && -n ${W4_ARM_GROUP_FILE:-} ]] || { echo "owned mode only under arm_supervisor.py"; exit 2; }
for i in {1..40}; do [[ -s $W4_ARM_GROUP_FILE ]] && break; sleep 0.05; done
"$W45R_PY" -I - "$W4_ARM_GROUP_FILE" <<'PY' || exit 2
import json, os, sys
r = json.load(open(sys.argv[1]))
assert r["supervisor_pid"] == int(os.environ["W4_SUPERVISOR_PID"])
assert r["pgid"] == os.getpgrp() and r["sid"] == os.getsid(0) and r["setsid_calls"] == 1
PY
W45R_LOG=$W45R_RUN/entry.log
export W45R_SESSION_DEADLINE=$(( W45R_ENTRY_DEADLINE - POST_RESERVE_S ))
[[ $ROLE == *rw3 ]] && export W45R_SLOTS=$TABLE || export W45R_ARMS=$TABLE
w45_log "OWNED $ENTRY_NAME mode $W45R_MODE pgid $(ps -o pgid= $$ | tr -d ' ') cgroup $(awk -F: '$1=="0"{print $3}' /proc/self/cgroup) session deadline $(date -u -d @"$W45R_SESSION_DEADLINE" +%FT%TZ)"
RC=0
if [[ $W45R_MODE == gpu ]]; then
  if w45_quiet; then echo QUIET; exit 3; fi
  bash "$W45R_GPU_LOCK" --timeout 120 --label v45-rc -- bash "$SESSION" > "$W45R_RUN/gpu_session.log" 2>&1 || RC=$?
else
  bash "$SESSION" > "$W45R_RUN/gpu_session.log" 2>&1 || RC=$?
fi
w45_log "session rc=$RC (GPU lock released)"
PRC=0
source "$W45R_ENVFILE"
export W45R_PINNED_SOURCE=$W45R_SNAP W45R_PIN_MANIFEST=$W45R_MANIFEST_PATH
export W45R_PRIVATE_ATLAS=<USER_HOME>/wrf_gpu2_lanes/v45-rc/smoke_timing_missing_v1/private_atlas
if [[ $POST == e41 && ( -n "$(ls -A "$W45R_P/jax/aot" 2>/dev/null)" || $W45R_MODE == smoke ) ]]; then
  env W45R_RUN="$([[ $W45R_MODE == gpu ]] && echo "$W45R_DATA_SCRATCH" || echo "$W45R_SMOKE_ROOT")/session_${W45R_PIN:0:12}_${W45R_MODE}_${W45R_SMOKE_SESSION_TAG:-v6}" bash "$W45R_TOOLS/e41_post.sh" > "$W45R_RUN/e41_post.out" 2>&1 || PRC=$?
  w45_log "E41 post rc=$PRC"
elif [[ $POST == profile ]]; then
  bash "$W45R_TOOLS/profile_post.sh" > "$W45R_RUN/profile_post.out" 2>&1 || PRC=$?
  w45_log "RW1 profile post rc=$PRC (separate from unprofiled RW timings)"
elif [[ $POST == rw3summary ]]; then
  bash "$W45R_TOOLS/rw3_summary.sh" > "$W45R_RUN/rw3_summary.out" 2>&1 || PRC=$?
  w45_log "RW3 summary rc=$PRC"
fi
LEFT=
if [[ $W45R_MODE == gpu ]]; then   # nothing of ours may stay on the GPU (sequencer contract)
  for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [[ $(ps -o sid= "$gp" 2>/dev/null | tr -d ' ') == "$(ps -o sid= $$ | tr -d ' ')" ]] && LEFT+="$gp "
  done
fi
for f in session.log post.log entry.log gpu_session.log e41_post.out rw3_summary.out seal.log; do
  [ -s "$W45R_RUN/$f" ] && cp "$W45R_RUN/$f" "$W45R_RECEIPTS/"; done
"$W45R_PY" -I - "$W45R_RUN/ENTRY_RESULT.json" "$RC" "$PRC" "$LEFT" <<'PY'
import json, sys
json.dump(dict(session_rc=int(sys.argv[2]), post_rc=int(sys.argv[3]), own_gpu_pids_left=sys.argv[4].split()), open(sys.argv[1], "w"), indent=1)
PY
cp "$W45R_RUN/ENTRY_RESULT.json" "$W45R_RECEIPTS/"
w45_log "ENTRY $ENTRY_NAME done session_rc=$RC post_rc=$PRC gpu_pids_left='$LEFT'"
[[ -z $LEFT ]] || exit 125
(( RC != 0 )) && exit "$RC"
exit "$PRC"
