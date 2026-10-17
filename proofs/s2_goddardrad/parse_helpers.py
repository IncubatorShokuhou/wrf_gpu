"""Stage compact, finite pristine REAL4 savepoints from helpers_driver stdout."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

build, destination = map(Path, sys.argv[1:3])
data = {k: np.full(shape, np.nan, np.float32) for k, shape in {
    'dz': (16,8), 'q': (16,8,6), 're': (16,8,6),
    'sw': (16,8,11,3), 'lw': (16,8,10,3),
    'latitude': (7,), 'day': (4,), 'climate': (7,4,33,6),
    'pressure': (8,), 'ozone': (7,4,8),
}.items()}
data['day'][:] = [1,30,212,365]
for line in (build/'helpers.txt').read_text().splitlines():
    words = line.split()
    if not words:
        continue
    name = words[0]
    if name == 'input':
        i,k = [int(x)-1 for x in words[1:3]]
        v = np.asarray(words[3:],np.float32)
        data['dz'][i,k],data['q'][i,k,:],data['re'][i,k,:] = v[0],v[1:7],v[7:13]
    elif name in ('sw','lw'):
        i,k,b = [int(x)-1 for x in words[1:4]]
        data[name][i,k,b,:] = np.asarray(words[4:],np.float32)
    elif name == 'climate':
        c,d,k = [int(x)-1 for x in words[1:4]]
        v = np.asarray(words[4:],np.float32)
        data['latitude'][c],data['climate'][c,d,k,:] = v[0],v[1:]
    elif name == 'ozone':
        c,d,k = [int(x)-1 for x in words[1:4]]
        v = np.asarray(words[4:],np.float32)
        data['latitude'][c],data['pressure'][k],data['ozone'][c,d,k] = v
assert all(np.isfinite(v).all() for v in data.values())
destination.mkdir(parents=True,exist_ok=True)
out = destination/'helpers_real4.npz'
np.savez_compressed(out, **data)
manifest = json.loads((build/'manifest.json').read_text())
manifest.update({
    'schema':'wrf-goddard-helpers-real4-v1',
    'savepoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'raw_sha256':hashlib.sha256((build/'helpers.txt').read_bytes()).hexdigest(),
    'shapes':{k:list(v.shape) for k,v in data.items()},
    'active_species_counts':np.count_nonzero(data['q'],axis=(0,1)).tolist(),
    'tolerances':{'optics_rtol':4e-6,'optics_atol':1e-10,'climate_rtol':5e-7,'climate_atol':1e-12,'ozone_rtol':1e-6,'ozone_atol':1e-14},
})
(destination/'helpers_real4.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
