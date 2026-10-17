"""Census a completed lane-owned original-WRF reference without changing it."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from netCDF4 import Dataset

CASE = Path(sys.argv[1]).resolve()
plan = json.loads((CASE/'PLAN.json').read_text())
assert 'SUCCESS COMPLETE WRF' in (CASE/'rsl.error.0000').read_text()
fields = ('QHAIL','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP','QNICE','QNRAIN',
          'QNCLOUD','QNDROP','QNSNOW','QNGRAUPEL','QNHAIL','QNCCN','QVGRAUPEL','QVHAIL','QNWFA','QNIFA')
receipt = {'status':'COMPLETE','original_cpu_wrf':True,'reference_plan':plan,'frames':{},'input_sha_unchanged':{}}
for name, metadata in plan['input_files'].items():
    receipt['input_sha_unchanged'][name] = hashlib.sha256((CASE/name).read_bytes()).hexdigest()==metadata['sha256']
assert all(receipt['input_sha_unchanged'].values())
for domain, frames in (('d01',3),('d02',7)):
    files = sorted(CASE.glob(f'wrfout_{domain}_*'))
    assert len(files)==frames, (domain,len(files))
    receipt['frames'][domain] = []
    for path in files:
        row = {'file':path.name,'bytes':path.stat().st_size,'fields':{}}
        with Dataset(path) as ds:
            for name in fields:
                if name not in ds.variables:
                    continue
                array = np.asarray(ds[name][0])
                assert np.isfinite(array).all(), (path,name)
                row['fields'][name] = {'nonzero':int(np.count_nonzero(array)),
                    'min':float(array.min()),'max':float(array.max()),
                    'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
        receipt['frames'][domain].append(row)
(CASE/'REFERENCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'case':str(CASE),'frames':{k:len(v) for k,v in receipt['frames'].items()},
                 'final_fields':{k:v[-1]['fields'] for k,v in receipt['frames'].items()}},indent=2))
