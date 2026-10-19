#!/usr/bin/env bash
# w42-bundle042: switch the PREPARED (not yet primed) v0.4.2 bundle session to a new BUNDLE PIN (manager re-pin), CPU only.
# usage: bash repin.sh NEW_PIN [--check]        (--check: print the plan, change nothing)
#  1. OLD pin from the current frozen tools (newest $L/tools_session_*/pin.env); NEW must be main (main:src == NEW src);
#     refuses if P is primed (P/jax/aot exists) or a GPU session of this lane runs
#  2. stops the OLD trigger (exact PID of $F_OLD/trigger.sh, cmdline verified)
#  3. P/src -> NEW (git checkout --detach in the P/src worktree; must be clean)
#  4. pin_prepare.sh NEW (RELID_CORES=25 AOTKEY_CORES=28 SKIP_RC=1): fresh frozen tools + PREREG, setup.sh, snapshot,
#     release identities Swiss + PROD, aot-key pairs, NEW trigger
#  5. RC delta OLD..NEW (rc_delta.py: importers of what changed; env RC_FILE_LIST=<file> replaces that list, e.g. the
#     manager-chosen subset for a version-string re-pin) on the new snapshot, workers on 8,9,12 (+13 until 12:20Z)
#  env TRIGGER_GO (e.g. 'GPU GO2') is passed to the new trigger (exact GO phrase)
#  6. old snapshot removed by exact path
set -euo pipefail
NEW=$1 CHECK=${2:-}
REPO=<USER_HOME>/src/wrf_gpu2; L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2; LANE=<DATA_ROOT>/wrf_gpu2/worktrees/w42-bundle042
T=$LANE/proofs/w42_bundle042/tools
F_OLD=$(ls -dt "$L"/tools_session_* | head -n 1); source "$F_OLD/pin.env"; OLD=$PIN; OLD12=${OLD:0:12}
NEW=$(git -C "$REPO" rev-parse "$NEW^{commit}"); NEW12=${NEW:0:12}; NTREE=$(git -C "$REPO" rev-parse "$NEW:src")
echo "old pin $OLD12 (frozen $F_OLD) -> new pin $NEW12 (src ${NTREE:0:12})"
[ "$OLD" != "$NEW" ] || { echo "same pin: nothing to do"; exit 0; }
[ "$(git -C "$REPO" rev-parse main:src)" = "$NTREE" ] || { echo "main:src != new pin src"; exit 4; }
echo "src files changed old..new:"; git -C "$REPO" diff --name-only "$OLD" "$NEW" -- src | sed 's/^/  /'
[ ! -e "$P/jax/aot" ] || { echo "P is primed (P/jax/aot): a re-pin needs a fresh bundle - ask the manager"; exit 4; }
if pgrep -f "tools_session_$OLD12/(gpu_session|session|rw3t).sh" > /dev/null; then echo "GPU session of the old pin runs: stop"; exit 4; fi
TRIG=""
for p in $(pgrep -f "tools_session_$OLD12/trigger.sh" || true); do
  /usr/bin/grep -q -a "tools_session_$OLD12/trigger.sh" "/proc/$p/cmdline" && TRIG="$TRIG $p"; done
echo "old trigger pid(s):${TRIG:- none}"
[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src)" ] || { echo "P/src dirty"; exit 4; }
# no `| head` under pipefail: a long selection SIGPIPEs rc_delta and set -e would abort here (re-pin b979188128: 1005 files)
python3 "$T/rc_delta.py" "$REPO" "$OLD" "$NEW" "$L/rc/repin_${OLD12}_${NEW12}_files.txt" > "$L/rc/repin_${OLD12}_${NEW12}_delta.log"
head -n 1 "$L/rc/repin_${OLD12}_${NEW12}_delta.log"
if [ -n "${RC_FILE_LIST:-}" ]; then   # manager-chosen subset replaces the importer delta (kept beside it for the record)
  mv "$L/rc/repin_${OLD12}_${NEW12}_files.txt" "$L/rc/repin_${OLD12}_${NEW12}_importers.txt"
  cp "$RC_FILE_LIST" "$L/rc/repin_${OLD12}_${NEW12}_files.txt"
  echo "RC_FILE_LIST: $(wc -l < "$RC_FILE_LIST") files (importer delta $(wc -l < "$L/rc/repin_${OLD12}_${NEW12}_importers.txt") kept as *_importers.txt)"
fi
if [ "$CHECK" = --check ]; then echo "CHECK ONLY: nothing changed"; exit 0; fi
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
for p in $TRIG; do kill -TERM "$p"; done
[ -z "$TRIG" ] || echo "old trigger stopped:$TRIG"
git -C "$P/src" checkout -q --detach "$NEW"
[ "$(git -C "$P/src" rev-parse HEAD)" = "$NEW" ] && [ "$(git -C "$P/src" rev-parse HEAD:src)" = "$NTREE" ]
echo "P/src -> $NEW12"
cd "$L" && RELID_CORES=25 AOTKEY_CORES=28 SKIP_RC=1 bash "$T/pin_prepare.sh" "$NEW" auto 8,9,12 all
SN=$S/snap_$NEW12; F=$L/tools_session_$NEW12; D=$L/rc/delta_$NEW12
python3 "$F/rc_suite.py" plan --tree "$SN" --out "$D" --file-list "$L/rc/repin_${OLD12}_${NEW12}_files.txt" \
  --baseline <USER_HOME>/wrf_gpu2_lanes/w41-knownred/FULL_FILE_RESULTS.json
python3 - "$D/plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for r in p["files"]:
    w = r["base_wall_s"]
    if r["file"] == "tests/test_v013_operational_smoke.py":
        r["tier"] = "20G"  # one process per file needs > 12 GiB (15.7 GiB measured on 0db963e321)
    r["timeout_active_s"] = 1800 if w is None else int(min(14400, max(900, 10 * w)))
json.dump(p, open(sys.argv[1], "w"), indent=2)
PY
bash "$F/rc_launch.sh" "$SN" "$D" "rA:8:20G:20G,12G,6G" "rB:9:6G:6G" "rC:12:6G:6G"
if [ "$(date -u +%H%M)" -lt 1200 ]; then
  setsid bash -c "exec '$LANE/scripts/with_ram_slot.sh' --label 'w42-bundle042-rcrD' --max 6G --timeout 28800 -- env JAX_PLATFORMS=cpu \
    GPUWRF_JAX_CACHE=0 taskset -c 13 nice -n 19 python3 '$F/rc_suite.py' worker --tree '$SN' --out '$D' --name rD --tiers 6G \
    --deadline 2026-10-19T12:15:00+00:00" > "$D/launch_rD.log" 2>&1 < /dev/null &
  echo "launched rD core 13 (stops claiming 12:15Z)"
fi
OSN=$S/snap_$OLD12
if [ -e "$OSN" ] && ! pgrep -f "snap_$OLD12" > /dev/null; then
  rm -f "$OSN/data/wrf_pristine" && git -C "$REPO" worktree remove "$OSN" && echo "old snapshot removed: $OSN"
fi
echo "REPIN_DONE old=$OLD12 new=$NEW12 delta=$D"
