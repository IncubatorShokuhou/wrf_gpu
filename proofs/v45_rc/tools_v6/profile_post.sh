#!/usr/bin/env bash
# CPU after GPU lock release; retain private reports until exact source/DAG binding is complete.
set -uo pipefail
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/lib.sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
source "$W45R_ENVFILE"
ROOT=$([[ $W45R_MODE == gpu ]] && echo "$W45R_DATA_SCRATCH" || echo "$W45R_SMOKE_ROOT")/session_${W45R_PIN:0:12}_${W45R_MODE}_${W45R_SMOKE_SESSION_TAG:-v6}
ARM=$ROOT/runs/${W45R_SMOKE_PROFILE_ARM:-B6_prod_profile}
umask 077
[ -s "$ARM/profile/roots.nsys-rep" ] || { echo 'PROFILE missing real report'; exit 4; }
timeout -k 30 600 nsys export --type sqlite --output "$ARM/profile/roots.sqlite" "$ARM/profile/roots.nsys-rep" > "$ARM/profile/export.log" 2>&1 || exit 4
JAX_PLATFORMS=cpu "$W45R_PY" "$W45R_TOOLS/profile_metrics.py" "$ARM/profile/roots.sqlite" "$ARM" "$ARM/profile/PROFILE_METRICS.json" "$W45R_MODE" || exit 5
JAX_PLATFORMS=cpu "$W45R_PY" -I - "$ARM/profile" <<'PY'
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
