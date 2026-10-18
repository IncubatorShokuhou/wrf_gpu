"""Summarise a PV/green-roof GPU smoke: finite history + EP_PV/QGR/TGR/DRAINGR per frame (CPU, no JAX).

python proofs/w41_writergaps/check_gpu_wrfout.py OUT_DIR RESULT.json
Proves finite execution and written fields only; the values are gated by the pristine oracle.
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

URBAN = ('EP_PV_URB3D', 'QGR_URB4D', 'TGR_URB4D', 'DRAINGR_URB3D')


def main():
    out, result = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(out.glob('wrfout_d01_*'))
    frames, nonfinite = [], {}
    for path in files:
        with netCDF4.Dataset(path) as ds:
            for name, var in ds.variables.items():
                if var.dtype.kind == 'f':
                    bad = int(np.count_nonzero(~np.isfinite(np.asarray(var[:]))))
                    if bad:
                        nonfinite[f'{path.name}:{name}'] = bad
            row = {'file': path.name, 'variables': len(ds.variables)}
            for name in URBAN:
                if name not in ds.variables:
                    row[name] = None
                    continue
                v = np.asarray(ds.variables[name][:], np.float64)
                row[name] = dict(min=float(v.min()), max=float(v.max()), nonzero=int(np.count_nonzero(v)))
            frames.append(row)
    status = ('PASS_GPU_SMOKE_ONLY' if files and not nonfinite
              and all(r[n] is not None for r in frames for n in URBAN)
              and any(r['EP_PV_URB3D']['nonzero'] for r in frames)
              and any(r['TGR_URB4D']['nonzero'] and r['DRAINGR_URB3D']['nonzero'] for r in frames[1:]) else 'FAIL')
    record = dict(status=status, frames=frames, nonfinite=nonfinite,
                  scope='finite execution + PV/GR history written; values gated by the pristine oracle, not by this run')
    result.write_text(json.dumps(record, indent=1)+'\n')
    print(json.dumps({'status': status, 'files': len(files), 'nonfinite': len(nonfinite)}))
    return 0 if status != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
