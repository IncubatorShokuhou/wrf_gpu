"""Fresh main-kernel search before each actual composite experiment."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import sys

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
root=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep');ident=sys.argv[1]
row=json.loads((root/ident/'inputs.json').read_text());mp=row['options']['mp_physics']
pattern=re.compile(r'\b'+re.escape(ident)+r'\b|\bmp'+str(mp)+r'\b|\bmp_physics\W+'+str(mp)+r'\b',re.I)
main=Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel');found=[]
for name in ('FINDINGS.md','EXPERIMENTS.tsv','INBOX.md'):
    rows=[f'{name}:{i}:{line}' for i,line in enumerate((main/name).read_text().splitlines(),1) if pattern.search(line)]
    found.extend(rows[-20:])
report=dict(utc=datetime.now(timezone.utc).isoformat(),case=ident,main_search_pattern=pattern.pattern,
    source_commit=sys.argv[2],matches=found,
    difference='Full24h CLI composite against its complete matched original CPU-WRF reference, on this immutable source. Prior column/smoke arms are narrower. Existing actual-run receipts are refused by the launcher; repair retries must archive and disclose the changed source.')
(root/ident/'MAIN_SEARCH_BEFORE_GPU.json').write_text(json.dumps(report,indent=2)+'\n')
