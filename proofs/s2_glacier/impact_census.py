"""Count WRF glacier caller points in the frozen release inputs, without JAX."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def census(path):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        veg = np.asarray(ds['IVGTYP'][0])
        ice_category = int(getattr(ds, 'ISICE', 15))
        seaice = np.asarray(ds['SEAICE'][0]) if 'SEAICE' in ds.variables else np.zeros_like(veg)
        mask = veg == ice_category
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    shape=list(veg.shape), isice=ice_category, glacier_caller_count=int(mask.sum()),
                    glacier_cold_init_count=int((mask & (seaice <= 0)).sum()),
                    glacier_locations=np.argwhere(mask).tolist())


def main():
    root = Path(__file__).resolve().parents[2]
    paths = list((root/'examples/switzerland_d01').glob('wrfinput_d0?'))
    paths += list(Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725').glob('wrfinput_d0?'))
    issues = ('0102', '0118', '0120', '0123', '0218', '0220', '0225', '0227', '0303',
              '0407', '0408', '0502', '0503', '0601', '0608', '0613', '0614', '0615')
    missing = []
    for issue in issues:
        attempt = 2 if issue in ('0118', '0225', '0303') else 1
        case = Path(f'<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_2026{issue}_18z_a{attempt}/run/run')
        for domain in ('d01', 'd02', 'd03'):
            path = case/f'wrfinput_{domain}'
            if path.is_file():
                paths.append(path)
            else:
                missing.append(str(path))
    payload = dict(reference='WRF module_sf_noahmpdrv.F:1049 VEGTYP == ISICE_TABLE',
                   rows=[census(path) for path in paths], missing=missing,
                   cache='NONE; GPUWRF_JAX_CACHE=0; no model execution')
    target = Path(sys.argv[1])
    target.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(domains=len(payload['rows']), missing=len(missing),
                         glacier_domains=[(r['path'], r['glacier_caller_count'])
                                          for r in payload['rows'] if r['glacier_caller_count']]), indent=2))


if __name__ == '__main__':
    main()
