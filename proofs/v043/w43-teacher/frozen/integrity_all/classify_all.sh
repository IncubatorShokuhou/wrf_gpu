#!/usr/bin/env bash
# All-frames integrity verdict for one case (manager 21:08:31Z): merge the per-domain --all-frames JSONs, R4 cell-watch data on every
# frame (r4_eval.py --integrity merged), registered-class verdict via integ_classes.classify (unchanged). Usage: classify_all.sh CASE GPU_WRFOUT_DIR OUTDIR
set -euo pipefail
c=${1:?}; W=${2:?}; O=${3:?}; G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate; base=${c%%_icp*}; CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_$base/run/run
P="nice -n 19 env JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3"
$P - "$O" <<'Y'
import json, sys
from pathlib import Path
O = Path(sys.argv[1]); doms = {}
for d in ("d01", "d02", "d03"):
    doms[d] = json.load(open(O / f"{d}.json"))["domains"][d]
first = json.load(open(O / "d01.json")); first["domains"] = doms; first["pass"] = all(v["pass"] for v in doms.values())
(O / "integrity_all_72h.json").write_text(json.dumps(first, indent=1) + "\n")
Y
$P $G/r4_eval.py --cpu-dir $CPU --gpu-dir $W --out $O/r4_watch_all_72h.json --integrity $O/integrity_all_72h.json > $O/r4_all.log 2>&1
$P - "$O" <<'Y'
import json, sys
sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate")
from integ_classes import classify, compact
O = sys.argv[1]; v = classify(json.load(open(f"{O}/integrity_all_72h.json")), json.load(open(f"{O}/r4_watch_all_72h.json")))
json.dump(v, open(f"{O}/classes_all_72h.json", "w"), indent=1, default=str); print(compact(v))
Y
