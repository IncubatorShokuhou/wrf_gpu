#!/usr/bin/env bash
# CPU gates of lane w41-optcont, run in the lane's GPU-queue slot while no GPU arm runs (manager 22:32Z):
# release identities Swiss (default case) + PROD d01 (must stay EXACT) and whole-suite collect-only (0 errors).
# usage: proofs/w41_optcont/cpu_gates.sh OUTDIR     (6G RAM slot each, cores 2,18 nice 19 (manager core rule 22:58Z), ALISIOS opens refused)
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
OUT=$1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$OUT"
slot() { "$REPO/scripts/with_ram_slot.sh" --label w41-optcont --max 6G --timeout 900 -- \
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$REPO/src" taskset -c 2,18 nice -n 19 timeout 600 "$@"; }
G="$REPO/proofs/w41_optcont/identity_guarded.py"
slot python3 "$G" "$REPO/proofs/s1_port/release_identity.py" "$REPO" "$OUT/swiss_identity.json" > "$OUT/swiss.log" 2>&1
echo "swiss rc=$?"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
slot python3 "$G" "$REPO/proofs/s1_port/release_identity.py" "$REPO" "$OUT/prod_identity.json" \
  <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 > "$OUT/prod.log" 2>&1
echo "prod rc=$?"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
(cd "$REPO" && slot python3 -m pytest --collect-only -q -p no:cacheprovider --basetemp "$OUT/collect_tmp" > "$OUT/collect.log" 2>&1)
echo "collect rc=$?"; tail -2 "$OUT/collect.log"
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
want = {"swiss": "3356e24924a9", "prod": "196aed9f9973"}
got = {k: json.loads((out/f"{k}_identity.json").read_text())["arms"]["release"]["sha"] for k in want
       if (out/f"{k}_identity.json").is_file()}
verdict = {k: got.get(k, "")[:12] == v for k, v in want.items()}
print(json.dumps({"identity": got, "exact": verdict}))
(out/"CPU_GATES.json").write_text(json.dumps({"identity": got, "exact": verdict}, indent=1)+"\n")
PY
