"""Register authentic onboarding cases and inventory their active land regimes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual')
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
results = []
for name in ('sf3_madrid', 'urban_madrid'):
    case = ROOT / name
    inp = case / 'input'
    on = json.loads((inp / 'onboarding.json').read_text())
    with Dataset(inp / 'wrfinput_d01') as ds:
        lu = np.asarray(ds['LU_INDEX'][0])
        urban = lu == int(ds.getncattr('ISURBAN'))
        row = {'config': name, 'urban_points': int(urban.sum()),
               'soil_levels': len(ds.dimensions['soil_layers_stag']),
               'ZS': np.asarray(ds['ZS'][0]).tolist(),
               'landuse': str(ds.getncattr('MMINLU')),
               'lake_points': int(np.count_nonzero(ds['LAKEMASK'][:])) if 'LAKEMASK' in ds.variables else 0,
               'seaice_points': int(np.count_nonzero(ds['SEAICE'][:])) if 'SEAICE' in ds.variables else 0,
               'shape': list(lu.shape)}
    if name == 'sf3_madrid':
        assert row['soil_levels'] == 6 and row['lake_points'] == 0 and row['seaice_points'] == 0, row
    else:
        assert row['urban_points'] > 0, row
    cpu = case / 'cpu'
    cpu.mkdir(exist_ok=True)
    for item in (WRF / 'run').iterdir():
        if item.is_file() and item.name not in ('namelist.input', 'wrf.exe', 'real.exe', 'ndown.exe'):
            (cpu / item.name).symlink_to(item.resolve())
    for filename in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01'):
        (cpu / filename).symlink_to(inp / filename)
    (cpu / 'wrf.exe').symlink_to(ROOT / 'pristine/wrf.exe')
    for item in (ROOT / 'bl9/cpu').glob('qr_acr*.dat'):
        (cpu / item.name).symlink_to(item)
    manifest = {'schema': 'gpuqual-input-v1', 'config': name, 'hours': 24,
                'case': 'Madrid GFS 20261013_00z; original WPS/real.exe',
                'dt_s': on['plan']['time_step'], 'history_minutes': 60,
                'plan': on['plan'], 'edits': on['qualification_overrides'], 'activation': row,
                'input_sha256': {n: hashlib.sha256((inp / n).read_bytes()).hexdigest()
                                  for n in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01')},
                'onboarding_manifest': str(inp / 'onboarding.json')}
    (case / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results.append(row)
(ROOT / 'CITY_ACTIVATION.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
