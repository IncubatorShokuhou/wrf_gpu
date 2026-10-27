#!/usr/bin/env bash
# CPU after GPU lock release; retain private reports until exact source/DAG binding is complete.
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/lib.sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
source "$W44R_ENVFILE"
ROOT=$W44R_SCRATCH/session_${W44R_PIN:0:12}_${W44R_MODE}_v5
ARM=$ROOT/runs/B6_prod_profile
umask 077
[ -s "$ARM/profile/roots.nsys-rep" ] || { echo 'PROFILE missing real report'; exit 4; }
timeout -k 30 600 nsys export --type sqlite --output "$ARM/profile/roots.sqlite" "$ARM/profile/roots.nsys-rep" > "$ARM/profile/export.log" 2>&1 || exit 4
JAX_PLATFORMS=cpu "$W44R_PY" "$W44R_TOOLS/profile_metrics.py" "$ARM/profile/roots.sqlite" "$ARM" "$ARM/profile/PROFILE_METRICS.json" "$W44R_MODE" || exit 5
JAX_PLATFORMS=cpu "$W44R_PY" -I - "$ARM/profile" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);rows=[]
for name in ('roots.nsys-rep','roots.sqlite','PROFILE_METRICS.json'):
 f=p/name
 with f.open('rb') as h:d=hashlib.file_digest(h,'sha256').hexdigest()
 rows.append(dict(name=name,bytes=f.stat().st_size,sha256=d))
(p/'PROFILE_SHA.json').write_text(json.dumps(rows,indent=2)+'\n')
PY
# A numerical parser result is not source attribution; keep GPU raw0600 until that consumer is DONE.
