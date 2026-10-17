"""Freeze real Swiss inputs for an explicitly cropped operator test domain."""
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

SOURCE = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfinput_d01')
DEST = Path(__file__).resolve().parents[2] / 'tests/v040/numbers'
DEST.mkdir(parents=True, exist_ok=True)
with Dataset(SOURCE) as ds:
    y, x, ny, nx = 12, 12, 12, 12
    names = ('U', 'V', 'QVAPOR', 'MU', 'MUB', 'MAPFAC_M', 'MAPFAC_U', 'MAPFAC_V',
             'C1H', 'C2H', 'FNM', 'FNP', 'RDNW')
    a = {}
    for name in names:
        var = ds[name]
        value = np.asarray(var[0], np.float32)
        if value.ndim >= 2:
            ey = int(var.dimensions[-2] == 'south_north_stag')
            ex = int(var.dimensions[-1] == 'west_east_stag')
            value = value[..., y:y+ny+ey, x:x+nx+ex]
        a[name] = value
    a['dx'] = np.asarray(ds.DX)
    a['dy'] = np.asarray(ds.DY)
np.savez_compressed(DEST/'swiss_transport.npz', **a)
(DEST/'swiss_transport.json').write_text(json.dumps({
    'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'crop_yx': [y, x], 'mass_shape': list(a['QVAPOR'].shape), 'variables': names,
    'scope': 'Real Swiss input crop used as an independent complete operator domain; not a coupled forecast. '
             'Non-Thompson number profiles constructed from real QVAPOR and disclosed in the test.',
}, indent=2)+'\n')
