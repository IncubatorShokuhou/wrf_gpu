#!/usr/bin/env bash
# B1 v0.4.1 release benchmark trigger at the manager's source pin. usage: trigger.sh BASE    (BASE = lane scratch dir)
# Program = the manager's pin (manager 2026-10-18T08:23:19Z): needs BOTH the file B1_PIN_FILE (one full main commit sha)
# AND a MAILBOX line '<ts>Z manager -> w41-bdyrec: B1 PIN <sha>' after MAILBOX line B1_MAILBOX_FROM whose sha is a
# prefix of the file's (the last such line counts). The pin must be a commit on main with main:src == pin:src; any
# missing/mismatching piece = refuse and keep waiting (posted once per state); no pin by END - 1 h = NOT STARTED.
# Per pin: setup.py builds BASE/b1_<pin12> (fresh snapshot, guarded WN3 inputs, D6 case dirs, empty caches). Then the
# window start B1_START_UTC (13:00Z), a valid manager lease, a foreign QUIET flag to clear, the pin re-checked right
# before the lock; ONE lock session (session.sh re-checks pin + main:src under the lock), GPU START/END lines, and the
# CPU follow-up (E41 before any cleanup, D6 of par24, identities, S1 proxy/whole-run, WN3 throughput) on the cores in
# B1_FOLLOWUP_CPUS_FILE (manager-assigned; waits while missing) in one 12G RAM slot, follow-up START/END lines.
# Every terminal path (follow-up finished or failed, B1 NOT STARTED) ends with the exact line 'B1 FOLLOWUP DONE': the
# w41-swddirscreen CPU twin restarts on 8,9,12,13 only after it (manager 2026-10-18T08:42:33Z). A kill posts nothing.
# Every wait compares the wall clock (hibernation-safe, E60/E180); watchers use /usr/bin/grep (ugrep wrapper trap).
set -uo pipefail
BASE=$(readlink -f "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec
MAIN_REPO=<USER_HOME>/src/wrf_gpu2
M=${B1_MAILBOX:-$MAIN_REPO/.agent/kernel/MAILBOX.md}        # B1_MAILBOX/B1_PIN_FILE/B1_LEASE/B1_DRY_ARM: dry test only
PIN_FILE=${B1_PIN_FILE:-<USER_HOME>/wrf_gpu2_lanes/manager/B1_PIN}
LEASE=${B1_LEASE:-/tmp/wrf_gpu2_run_approvals/w41-bdyrec.lease}
CPUS_FILE=${B1_FOLLOWUP_CPUS_FILE:-$L/B1_FOLLOWUP_CPUS}
N0=${B1_MAILBOX_FROM:?MAILBOX line after which PIN lines count}
START=$(date -u -d "${B1_START_UTC:-2026-10-18T13:00:00Z}" +%s)
END=$(date -u -d "${B1_END_UTC:-2026-10-18T19:00:00Z}" +%s)
PIN_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w41-bdyrec: B1 PIN [0-9a-f]{7,40}'
TLOG=${B1_TRIGGER_LOG:-$L/b1_trigger.log}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$TLOG"; }
post() { echo "$(date -u +%FT%TZ) w41-bdyrec: $*" >> "$M"; log "posted: $*"; }
followup_done() { post "B1 FOLLOWUP DONE"; }   # exact text, watched by w41-swddirscreen
said=
say_once() {  # say_once KEY post|log TEXT: one message per distinct waiting/refusal state
  [[ "$said" == "$1" ]] && return 0
  said=$1; "$2" "$3"
}
pin_state() {  # sets PIN, ROOT and returns 0 when the pin is valid now, else sets WHY and returns 1
  local file line ptree mtree
  PIN=; ROOT=; WHY=
  file=$( { tr -d '[:space:]' < "$PIN_FILE"; } 2>/dev/null )
  [[ "$file" =~ ^[0-9a-f]{40}$ ]] || { WHY="wait: pin file $PIN_FILE missing/not one full sha ('${file:0:48}')"; return 1; }
  line=$(tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -E "$PIN_RE" | tail -1 | /usr/bin/grep -oE 'B1 PIN [0-9a-f]+' | cut -d' ' -f3)
  [[ -n "$line" ]] || { WHY="wait: no 'manager -> w41-bdyrec: B1 PIN <sha>' line after MAILBOX line $N0 (file has ${file:0:12})"; return 1; }
  [[ "$file" == "$line"* ]] || { WHY="REFUSE: pin file ${file:0:12} != MAILBOX PIN line $line"; return 1; }
  git -C "$MAIN_REPO" cat-file -e "$file^{commit}" 2>/dev/null || { WHY="REFUSE: pin $file is not a commit"; return 1; }
  git -C "$MAIN_REPO" merge-base --is-ancestor "$file" main || { WHY="REFUSE: pin ${file:0:12} is not on main"; return 1; }
  ptree=$(git -C "$MAIN_REPO" rev-parse "$file:src"); mtree=$(git -C "$MAIN_REPO" rev-parse main:src)
  [[ "$ptree" == "$mtree" ]] || { WHY="REFUSE: main src tree ${mtree:0:12} != pin ${file:0:12} src tree ${ptree:0:12}"; return 1; }
  PIN=$file; ROOT=$BASE/b1_${file:0:12}
}
lease_ok() { local exp; exp=$(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
  [[ -n "$exp" ]] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) > $(date +%s) + 1800 )); }

log "armed (tools $HERE): BASE $BASE; pin = $PIN_FILE + /$PIN_RE/ after MAILBOX line $N0; window $(date -u -d @$START +%FT%TZ)-$(date -u -d @$END +%FT%TZ)"
while :; do
  (( $(date +%s) < END - 3600 )) || { post "B1 NOT STARTED: no valid pin/lease/QUIET-free start before $(date -u -d @$((END - 3600)) +%FT%TZ) (last state: $said)."; followup_done; exit 1; }
  if ! pin_state; then
    case "$WHY" in REFUSE*) say_once "$WHY" post "B1 $WHY (waiting for a corrected pin)";; *) say_once "$WHY" log "$WHY";; esac
    sleep 60; continue
  fi
  if [ ! -e "$ROOT/PLAN.json" ]; then
    if [ -e /tmp/wrf_gpu2_quiet ]; then say_once "setupquiet$PIN" log "setup ${PIN:0:12} waits: QUIET $(cat /tmp/wrf_gpu2_quiet 2>/dev/null)"; sleep 30; continue; fi
    [ -e "$ROOT" ] && { post "B1 NOT STARTED: $ROOT exists without PLAN.json (failed setup), needs a look."; followup_done; exit 1; }
    log "setup pin $PIN -> $ROOT"
    if ! JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 nice -n 19 python3 "$HERE/setup.py" "$ROOT" "$PIN" >> "$L/b1_setup_${PIN:0:12}.log" 2>&1; then
      post "B1 NOT STARTED: setup at pin ${PIN:0:12} failed ($L/b1_setup_${PIN:0:12}.log)."; followup_done; exit 1
    fi
    log "setup done: $(tail -1 "$L/b1_setup_${PIN:0:12}.log" | cut -c1-300)"
    post "B1 pin ${PIN:0:12} accepted (file == MAILBOX line, on main, main:src == pin:src $(git -C "$MAIN_REPO" rev-parse "$PIN:src" | cut -c1-12)); snapshot + D6 dirs + empty caches built ($ROOT); waiting for $(date -u -d @$START +%TZ)."
  fi
  if (( $(date +%s) < START )); then say_once "start$PIN" log "pin ${PIN:0:12} ready; waiting for the window start"; sleep 30; continue; fi
  if ! lease_ok; then say_once "lease" post "B1 waiting: lease $LEASE missing/expires within 30 min ($(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null))."; sleep 60; continue; fi
  if [ -e /tmp/wrf_gpu2_quiet ]; then say_once "quiet" log "foreign QUIET: $(cat /tmp/wrf_gpu2_quiet 2>/dev/null)"; sleep 30; continue; fi
  break
done
TREE=$(python3 -c "import json; print(json.load(open('$ROOT/PLAN.json'))['src_tree'])")
WAIT=$(( END - 3600 - $(date +%s) ))   # lock queue wait; the session itself cuts every arm to 19:00Z - 2 min
post "GPU START B1 v0.4.1 release benchmark (pin $PIN, src $TREE; one lock session <= 150 GPU-min, ends before $(date -u -d @$END +%FT%TZ))"
LOCK=(<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w41-bdyrec --timeout "$WAIT" --)
[ -n "${B1_DRY_ARM:-}" ] && LOCK=()   # dry: session.sh sleeps instead of model work, no lock
"${LOCK[@]}" env B1_PIN_FILE="$PIN_FILE" bash "$HERE/session.sh" "$ROOT"
rc=$?
MIN=$(/usr/bin/grep -oE 'END( \(signal\))?( GPU part:)? [0-9]+ GPU-min' "$ROOT/session.log" 2>/dev/null | /usr/bin/grep -oE '[0-9]+' | tail -1)
post "GPU END B1 rc=$rc, ${MIN:-?} GPU-min (lock held); CPU follow-up (E41 on both caches, par24 nested D6, identities, S1 summaries, WN3 throughput) next."
said=
until CPUS=$( { tr -d '[:space:]' < "$CPUS_FILE"; } 2>/dev/null ) && [[ "$CPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; do
  say_once cpus post "B1 follow-up NEED: no valid core list in $CPUS_FILE; waiting."; sleep 60
done
while [ -e /tmp/wrf_gpu2_quiet ]; do say_once fquiet log "follow-up waits: QUIET $(cat /tmp/wrf_gpu2_quiet 2>/dev/null)"; sleep 30; done
post "B1 follow-up START on $CPUS nice 19, one 12G slot (ROOT $ROOT)."
taskset -c "$CPUS" nice -n 19 <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-bdyrec --max 12G --timeout 10800 -- \
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3 "$HERE/followup.py" "$ROOT" ${B1_DRY_ARM:+--skip-e41 --skip-ids} \
  > "${TLOG%.log}_followup.log" 2>&1
frc=$?
log "followup rc=$frc -> $ROOT/RESULT.json"
python3 - "$ROOT/RESULT.json" >> "$TLOG" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
g = lambda d, *k: (d if not k else g(d.get(k[0]) if isinstance(d, dict) else None, *k[1:]))  # noqa: E731
print("AUTO-SUMMARY", json.dumps({"S1_proxy": g(r, "S1_proxy", "s1_warm_s_per_fc_h"), "aa": g(r, "S1_proxy", "aa", "aa_floor_pct"),
      "S1_whole": g(r, "S1_whole", "s_per_fc_h"), "wn3_par24": g(r, "WN3", "par24", "wall_s_per_case_hour"),
      "D6": g(r, "D6", "verdicts"), "E41": {k: g(v, "gate") for k, v in (r.get("E41") or {}).items()},
      "identities": r.get("identities"), "core_gate": g(r, "core_gate", "arms")}))
EOF
post "B1 follow-up END rc=$frc (RESULT $ROOT/RESULT.json; numbers + disclosures follow from the lane)."
followup_done
