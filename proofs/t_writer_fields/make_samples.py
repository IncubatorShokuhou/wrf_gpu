"""Freeze small nonzero original-WRF samples for producer-to-writer seam tests."""
import hashlib
import json
from pathlib import Path
import sys
import netCDF4
import numpy as np

root, out = map(Path, sys.argv[1:3])
samples, receipt = {}, {}
for case, names in [('c08', ['EXCH_TKE']), ('c09', ['RIMI']), ('c45', [
        'RE_CLOUD_GSFC', 'RE_ICE_GSFC', 'RE_SNOW_GSFC',
        'RE_RAIN_GSFC', 'RE_GRAUPEL_GSFC', 'RE_HAIL_GSFC'])]:
    path = sorted((root / case / 'cpu').glob('wrfout_d01_*'))[-1]
    with path.open('rb') as f:
        file_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    with netCDF4.Dataset(path) as ds:
        for name in names:
            a = np.asarray(ds[name][0])
            _, j, i = np.unravel_index(np.argmax(a), a.shape)
            j, i = max(0, min(j, a.shape[-2]-4)), max(0, min(i, a.shape[-1]-4))
            v = a[:, j:j+4, i:i+4].copy()
            samples[name] = v
            receipt[name] = dict(file=str(path), file_sha256=file_sha,
                selection=[':', f'{j}:{j+4}', f'{i}:{i+4}'],
                data_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
                attrs={n: ds[name].getncattr(n) for n in ds[name].ncattrs()},
                min=float(v.min()), max=float(v.max()))
np.savez(out / 'producer_samples.npz', **samples)
(out / 'PRODUCER_SAMPLES.json').write_text(json.dumps(receipt, indent=2,
    default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')
print({n: (v.shape, float(v.max())) for n, v in samples.items()})
