#!/usr/bin/env bash
# CPU-only lifecycle proof of session.sh scope cleanup (critic cr-w41-bdyrec 00:10:58Z): no GPU, no model.
# usage: lifecycle_proof.sh TOOLS_DIR OUT_JSON   (TOOLS_DIR = the immutable copy that will be armed)
# Builds a dry ROOT (setup.py '-': snapshot + case dirs, no ALISIOS), then runs the REAL session.sh with
# REQUAL_DRY_ARM (each arm scope runs sleep) on three paths: normal end, TERM to the session while an arm scope
# runs, simulated session error (exit 7) while an arm scope runs. Invariant sampled every 0.2 s: the session process
# is never gone while a scope w41bdyrec-req-<session pid>-* is still active; no scope is left at the end.
set -uo pipefail
export LC_ALL=C   # decimal point in printf (German locale writes 17,0)
TOOLS=$1; OUT=$2
ROOT=<DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/composed_lifecycle
REPO=<USER_HOME>/src/wrf_gpu2_wt/w41-bdyrec
[ ! -e "$ROOT" ] || { echo "refuse: $ROOT exists"; exit 2; }
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3 "$TOOLS/setup.py" "$ROOT" "$(git -C "$REPO" rev-parse HEAD)" lane/w41-swddir 969baaba1c - \
  > /dev/null || { echo "dry setup failed"; exit 3; }
scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
  "w41bdyrec-req-$1-*" | awk '{print $1}'; }
run_case() {  # run_case NAME SIGNAL_AFTER_FIRST_SCOPE(0|1) ENV...
  local name=$1 signal=$2; shift 2
  env "$@" bash "$TOOLS/session.sh" "$ROOT" > "$ROOT/$name.out" 2>&1 & local spid=$!
  local seen=0 violations=0 samples=0 max_scopes=0 t0; t0=$(date +%s.%N)
  while :; do
    local alive=0 n; kill -0 "$spid" 2>/dev/null && alive=1
    n=$(scopes "$spid" | wc -l); samples=$((samples + 1)); (( n > max_scopes )) && max_scopes=$n
    (( n > 0 && seen == 0 )) && { seen=1; (( signal )) && kill -TERM "$spid"; }
    (( alive == 0 && n > 0 )) && violations=$((violations + 1))
    (( alive == 0 )) && break
    sleep 0.2
  done
  wait "$spid"; local rc=$?
  local left; left=$(scopes "$spid" | wc -l)
  local sleeps; sleeps=$(pgrep -fc "^sleep 117$" || true)
  printf '{"case": "%s", "rc": %s, "scope_seen": %s, "max_scopes": %s, "samples": %s, "violations": %s, "scopes_left": %s, "wall_s": %.1f, "dry_sleeps_alive": %s}\n' \
    "$name" "$rc" "$seen" "$max_scopes" "$samples" "$violations" "$left" "$(echo "$(date +%s.%N) - $t0" | bc)" "$sleeps"
}
{
  echo "["
  run_case normal 0 REQUAL_DRY_ARM=2; echo ","
  run_case signal 1 REQUAL_DRY_ARM=117; echo ","
  run_case error 0 REQUAL_DRY_ARM=117 REQUAL_DRY_EXIT_AFTER_START=1
  echo "]"
} > "$OUT"
cat "$OUT"
cp "$ROOT/session.log" "${OUT%.json}.session.log"
git -C "$REPO" worktree remove --force "$ROOT/snap" && rm -rf <DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/composed_lifecycle
