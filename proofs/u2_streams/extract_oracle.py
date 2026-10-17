"""Retain only small point values, metadata and hashes from actual auxhist7."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from gpuwrf.io.netcdf_lock import Dataset

root,out = (Path(p).resolve() for p in sys.argv[1:3])
out.mkdir(parents=True,exist_ok=True)
manifest = json.loads((root/'STAGING.json').read_text())
files = sorted(root.glob('auxhist7_d01_*'))
assert len(files) == 5,files
values = {name:[] for name in manifest['fields']}
times = []
for path in files:
    with Dataset(path) as ds:
        times.append(b''.join(ds['Times'][0]).decode())
        for name in values:
            raw = np.asarray(ds[name][0])
            values[name].append(raw[[19,11],[19,8]])
values = {key:np.asarray(value,np.float32) for key,value in values.items()}
np.savez(out/'frames.npz',**values)
manifest.update(raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                times=times,grid_points_1based=[[20,20],[9,12]],
                values_sha256=hashlib.sha256((out/'frames.npz').read_bytes()).hexdigest(),
                shape_per_field=[5,2],units={'T2':'K','Q2':'kg kg-1','U10':'m s-1','V10':'m s-1',
                    'PSFC':'Pa','LH':'W m-2','HFX':'W m-2','PBLH':'m','TSK':'K','RAINC':'mm','RAINNC':'mm',
                    'SWDOWN':'W m-2','GLW':'W m-2'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(files),'times':times,'bytes':(out/'frames.npz').stat().st_size}))
