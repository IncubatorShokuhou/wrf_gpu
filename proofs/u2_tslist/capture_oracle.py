"""Extract tiny station neighborhoods and unmodified TS files from a CPU-WRF run."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

from netCDF4 import Dataset
import numpy as np

source, target = map(Path, sys.argv[1:3])
target.mkdir(parents=True, exist_ok=True)
fields = ('U', 'V', 'W', 'PH', 'PHB', 'P', 'PB', 'THM', 'QVAPOR', 'QCLOUD', 'QRAIN',
          'QICE', 'QSNOW', 'QGRAUP', 'T2', 'Q2', 'U10', 'V10', 'PSFC', 'GLW', 'SWDOWN', 'SWUPB',
          'HFX', 'LH', 'TSK', 'TSLB', 'RAINC', 'RAINNC', 'COSALPHA', 'SINALPHA')
files = sorted(source.glob('wrfout_d01_*'))
assert len(files) == 5, files
collected = {key: [] for key in fields}
points = [(19, 19), (8, 11)]  # x,y: same locations as stage_oracle, zero-based.
for path in files[1:]:
    with Dataset(path) as ds:
        for key in fields:
            # Tiny spatial patches include the source's +i/+j wind-rotation operands.
            patches = [np.asarray(ds[key][0, ..., y:y+2, x:x+2]) for x, y in points]
            collected[key].append(np.stack(patches))
        znw = np.asarray(ds['ZNW'][0], np.float32)
        attrs = {key: ds.getncattr(key) for key in ('MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', 'DX')}
with Dataset(source / 'wrfinput_d01') as ds:
    meta = {key: np.asarray(ds[key][0]) for key in ('XLAT', 'XLONG', 'HGT')}
np.savez_compressed(target / 'columns.npz', **{key: np.stack(values) for key, values in collected.items()},
                    ZNW=znw, **meta)
for path in sorted(source.glob('*.d01.*')):
    shutil.copyfile(path, target / path.name)
shutil.copyfile(source / 'tslist', target / 'tslist')
receipt = json.loads((source / 'STAGING.json').read_text())
receipt.update(projection=attrs, raw_history_files=[str(p) for p in files],
               raw_history_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               fields=fields, points_zero_based=points,
               fixture_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in target.iterdir() if p.is_file() and p.name != 'manifest.json'},
               oracle='Original CPU-WRF, no GPU code; 4 completed own steps at t18/36/54/72 seconds')
(target / 'manifest.json').write_text(json.dumps(receipt, indent=2, default=lambda v: v.item()) + '\n')
