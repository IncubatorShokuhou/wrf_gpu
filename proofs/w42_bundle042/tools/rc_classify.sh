#!/usr/bin/env bash
# w42-bundle042: classify the RC-suite failures (contract: every F pre-existing on the v0.4.1 tag, or new).
# usage: bash rc_classify.sh RC_OUT [CORE_TAG=8] [CORE_RERUN=9]          then, when both workers are done:
#        bash rc_classify.sh RC_OUT --compare
# Phase 1: rc_suite.py report on RC_OUT; every failing file that exists at v0.4.1 is re-run with the SAME runner/env on the
# tag snapshot (RC_OUT/../tag041_<name>); files killed by the runner timeout (rc -9, INT+KILL) are also re-run on the
# candidate snapshot (RC_OUT/../rerun_<name>) with timeout_active 3600 s (both plans). Files absent at the tag = NEW_TEST.
# Phase 2 (--compare): rc_compare.py candidate-vs-tag; for killed files the candidate re-run result replaces the kill.
set -euo pipefail
OUT=$(cd "$1" && pwd); T=$(cd "$(dirname "$0")" && pwd); NAME=$(basename "$OUT")
TAG=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/snap_v041; TO=$(dirname "$OUT")/tag041_$NAME; RO=$(dirname "$OUT")/rerun_$NAME
BASE=<USER_HOME>/wrf_gpu2_lanes/w41-knownred/FULL_FILE_RESULTS.json
if [ "${2:-}" = --compare ]; then
  python3 "$T/rc_compare.py" "$OUT" "$TO" "$OUT/CLASSIFY_vs_tag.json"
  [ -s "$RO/plan.json" ] && python3 "$T/rc_suite.py" report --out "$RO" && python3 "$T/rc_compare.py" "$RO" "$TO" "$OUT/CLASSIFY_rerun_vs_tag.json"
  exit 0
fi
CT=${2:-8} CR=${3:-9}
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
CAND=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tree'])" "$OUT/plan.json")
python3 "$T/rc_suite.py" report --out "$OUT"
python3 - "$OUT" "$TAG" "$TO.files" "$RO.files" <<'PY'
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), Path(sys.argv[2])
rep = json.loads((out / "report.json").read_text())
tagf = sorted({f["file"] for f in rep["failing_files"] if (tag / f["file"]).exists()})
rerun = sorted({f["file"] for f in rep["failing_files"] if f["killed"]})
Path(sys.argv[3]).write_text("".join(x + "\n" for x in tagf))
Path(sys.argv[4]).write_text("".join(x + "\n" for x in rerun))
print("tag re-run files", len(tagf), "| killed -> candidate re-run", len(rerun),
      "| NEW_TEST (absent at v0.4.1):", sorted({f["file"] for f in rep["failing_files"] if not (tag / f["file"]).exists()}))
PY
plan() {  # plan TREE OUTDIR FILELIST
  [ -s "$3" ] || return 1
  python3 "$T/rc_suite.py" plan --tree "$1" --out "$2" --file-list "$3" --baseline "$BASE"
  python3 - "$2/plan.json" "$RO.files" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); killed = set(open(sys.argv[2]).read().split())
for r in p["files"]:
    if r["file"] in killed:
        r["timeout_active_s"] = 3600
json.dump(p, open(sys.argv[1], "w"), indent=2)
tiers = {r["tier"] for r in p["files"]}
print("TIERS", ",".join(t for t in ("20G", "12G", "6G") if t in tiers or t == "6G"))
PY
}
spec() {  # spec NAME CORE PLANDIR -> NAME:CORE:MAX:TIERS covering every tier in the plan
  local tiers; tiers=$(python3 -c "import json,sys; t={r['tier'] for r in json.load(open(sys.argv[1]))['files']}; print(','.join(x for x in ('20G','12G','6G') if x in t or x == '6G'))" "$3/plan.json")
  echo "$1:$2:${tiers%%,*}:$tiers"
}
if plan "$TAG" "$TO" "$TO.files"; then bash "$T/rc_launch.sh" "$TAG" "$TO" "$(spec tA "$CT" "$TO")"; fi
if plan "$CAND" "$RO" "$RO.files"; then bash "$T/rc_launch.sh" "$CAND" "$RO" "$(spec rA "$CR" "$RO")"; fi
echo "CLASSIFY_LAUNCHED tag=$TO rerun=$RO (then: bash $0 $OUT --compare)"
