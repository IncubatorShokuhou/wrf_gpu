"""Validate tracked source/oracle assets from a clean git archive, without JAX."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import sys
import numpy as np
root=Path(__file__).resolve().parents[2]
commit=sys.argv[1] if len(sys.argv)>1 else 'HEAD'
paths=['proofs/s1_mpfix','tests/savepoint/test_s1_mpfix_morrison_dry.py',
       'tests/savepoint/test_s1_mpfix_morrison_adapter.py',
       'tests/v025/b_column/test_s1_mpfix_radii.py','tests/v025/b_column/test_s1_mpfix_full_radiation.py',
       'tests/v034/nssl2mom/test_radius_handoff.py']
raw=subprocess.check_output(['git','archive',commit,*paths],cwd=root)
files={}
with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
    for item in archive:
        if item.isfile():files[item.name]=archive.extractfile(item).read()
fixtures={n:json.loads(data) for n,data in files.items() if n.startswith('proofs/s1_mpfix/fixtures/') and n.endswith('.json')}
qualification=json.loads(files['proofs/s1_mpfix/QUALIFICATION.json'])
assert qualification['status']=='READY_FOR_CRITIC'
assert qualification['nested_admitted_mp']==[]
for name in ('SOURCE_SEAL.json','NESTED_ADMISSION.json','EXTERNAL_GPU_RECEIPTS.json'):
    assert f'proofs/s1_mpfix/{name}' in files
assert len(fixtures)==15,len(fixtures)
for mode in ('fp32','fp64'):
    for case in range(8,13):
        data=fixtures[f'proofs/s1_mpfix/fixtures/dry_{mode}_{case}.json']
        for values in data['columns'].values():assert np.isfinite(values).all()
        nc=np.array(data['columns']['NC_OUT'])
        assert (np.count_nonzero(nc)==0 if case in (8,9) else np.count_nonzero(nc)==40)
for name in ('thompson_aero_radii','radiation_mediation','radius_manifest','full_radiation_driver'):
    assert f'proofs/s1_mpfix/fixtures/{name}.json' in fixtures
full=fixtures['proofs/s1_mpfix/fixtures/full_radiation_driver.json']
assert len(full['rows'])==18
assert full['caller_controls']['swint_opt']==0 and full['caller_controls']['p_top']==5000.
for row in full['rows']:
    assert np.isfinite(row['scalars']).all() and np.isfinite(row['profiles']).all()
print(json.dumps(dict(status='PASS',commit=subprocess.check_output(['git','rev-parse',commit],cwd=root,text=True).strip(),
 files=len(files),fixtures=len(fixtures),archive_sha256=hashlib.sha256(raw).hexdigest(),
 hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}),indent=2))
