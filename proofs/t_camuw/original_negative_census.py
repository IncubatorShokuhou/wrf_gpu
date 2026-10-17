"""Read-only negative-value census of retained original-WRF hourly outputs.

Hourly observations cannot establish equivalence to 800 internal-stage visits.
Load the production NetCDF lock directly, without starting another JAX process.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root, reference, output = (Path(x).resolve() for x in sys.argv[1:4])
spec = importlib.util.spec_from_file_location("wrf_netcdf_lock", root / "src/gpuwrf/io/netcdf_lock.py")
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)
assert "jax" not in sys.modules
records = []
names = dict(qv="QVAPOR", qc="QCLOUD", qr="QRAIN", qi="QICE", qs="QSNOW", qg="QGRAUP",
             Ni="QNICE", Nr="QNRAIN")
sources = {}
for hour, path in enumerate(sorted(reference.glob("wrfout_d01_*"))[:5]):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with lock.Dataset(path) as data:
        data.set_auto_mask(False)
        for field, name in names.items():
            if name not in data.variables:
                records.append(dict(hour=hour, source=str(path), field=field, coverage="UNAVAILABLE"))
                continue
            value = np.asarray(data[name][0])
            negative = np.isfinite(value) & (value < 0)
            ny, nx = value.shape[-2:]
            j, i = np.ogrid[:ny, :nx]
            ring = (j == 0) | (j == ny - 1) | (i == 0) | (i == nx - 1)
            count = int(negative.sum())
            record = dict(hour=hour, source=str(path), field=field, dtype=str(value.dtype),
                          shape=list(value.shape), negative_count=count,
                          ring0_count=int((negative & ring).sum()),
                          interior_count=int((negative & ~ring).sum()),
                          nonfinite_count=int((~np.isfinite(value)).sum()))
            if count:
                index = np.unravel_index(np.argmin(value), value.shape)
                record["minimum"] = dict(value=float(value[index]), index_kji=list(map(int, index)))
            records.append(record)
output.write_text(json.dumps(dict(reference="original CPU-WRF retained outputs; read-only",
    coverage="hourly endpoints h0..4, not internal-stage or every-step coverage",
    source_sha256=sources, records=records), indent=2) + "\n")
print(json.dumps([r for r in records if r.get("negative_count")], indent=2))
