"""Place new bulky arm products on owner-approved data scratch, not ROOT."""
from pathlib import Path
import json
import os
import re
import shutil
import sys

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ident=sys.argv[1]
assert re.fullmatch(r'c\d\d',ident) or ident=='WN3-smoke'
base=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms')/ident
case=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/ident
assert shutil.disk_usage('<DATA_ROOT>').free >= 120*1024**3, 'Owner120GiB data floor'
paths={}
for name in ('cache','tmp','gpu_scratch','gpu','gpu_proof','e41'):
    target=base/name;link=case/name
    target.mkdir(parents=True,exist_ok=True)
    if link.is_symlink():assert link.resolve()==target
    else:
        assert not link.exists(), f'Preserve existing arm before retry: {link}'
        link.symlink_to(target,target_is_directory=True)
    paths[name]=dict(logical=str(link),physical=str(target))
(case/'SPILL_PATHS.json').write_text(json.dumps(dict(paths=paths,
    authority='Manager2026-10-14T14:29:15Z',data_free_bytes=shutil.disk_usage('<DATA_ROOT>').free),indent=2)+'\n')
