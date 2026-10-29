#!/usr/bin/env bash
# Item3 schema2/native prescreens already measured on this source. Consume the exact proof; no redundant compiles.
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/lib.sh
exec env JAX_PLATFORMS=cpu "$W45R_PY" -I - "${1:?manifest}" <<'PY'
import json,sys
from pathlib import Path
m=json.load(open(sys.argv[1]));p=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc/receipts')/('identity_'+m['pin'][:12])/'IDENTITY_BRANCH.json';j=json.load(p.open());assert j['pin']==m['pin'] and j['swiss_branch']=='b_changed_program';print('IDENTITY_READY',p,'Swiss24h originalCPU D6 mandatory')
PY
