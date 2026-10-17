"""Oracle input columns for refl10cm_hm from original CPU-WRF c08 (mp10) / c20 (mp40) history.

usage: python make_inputs.py <out_dir>   -> inputs.npz, input_fp32.bin, input_fp64.bin
Columns: per case 75 melting-layer columns (T>273.15 rain under snow/graupel) + 75 other active
columns (deterministic stride over all frames), plus synthetic threshold/melting-branch columns.
Temperature = (T+300)*(P/1e5)**(2/7) from the history state (any realistic REAL input suffices:
the oracle evaluates the pristine routine on exactly these inputs).
"""
import glob, hashlib, sys
from pathlib import Path
import numpy as np, netCDF4

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
NAMES = ('qv', 'qr', 'nr', 'qs', 'ns', 'qg', 'ng', 't', 'p')
cols, provenance = [], []
for case in ('c08', 'c20'):
    melt_cols, other_cols = [], []
    for f in sorted(glob.glob(f'<USER_HOME>/wrf_gpu2_lanes/v-prep/{case}/cpu/wrfout_d01_*')):
        with netCDF4.Dataset(f) as ds:
            g = lambda n: np.asarray(ds[n][0], np.float64)
            p = g('P') + g('PB'); t = (g('T') + 300.) * (p / 1e5) ** (2. / 7.)
            fields = dict(qv=g('QVAPOR'), qr=g('QRAIN'), nr=g('QNRAIN'), qs=g('QSNOW'), ns=g('QNSNOW'),
                          qg=g('QGRAUP'), ng=g('QNGRAUPEL'), t=t, p=p)
            qr, qs, qg = fields['qr'], fields['qs'], fields['qg']
            act = ((qr > 1e-9) | (qs > 1e-9) | (qg > 1e-9)).any(0)
            melt = ((t[:-1] > 273.15) & (qr[:-1] > 1e-9) & ((qs[1:] > 1e-9) | (qg[1:] > 1e-9))).any(0)
            for (j, i) in np.argwhere(melt):
                melt_cols.append((f, j, i, np.stack([fields[n][:, j, i] for n in NAMES])))
            for (j, i) in np.argwhere(act & ~melt)[::97]:
                other_cols.append((f, j, i, np.stack([fields[n][:, j, i] for n in NAMES])))
    for group, kind in ((melt_cols, 'melt'), (other_cols, 'active')):
        idx = np.linspace(0, len(group) - 1, 75).astype(int)
        for n in idx:
            f, j, i, a = group[n]
            cols.append(a); provenance.append(f'{case}:{Path(f).name}:{j}:{i}:{kind}')
base = cols[0].copy()
for label, edit in (('qr_at_threshold', dict(qr=1e-9, nr=1e3)), ('qs_just_above', dict(qs=np.nextafter(np.float32(1e-9), np.float32(1)), ns=1e4)),
                    ('melting_snow_top', None), ('melting_graupel_top', None)):
    a = base.copy()
    if edit:
        for k, v in edit.items(): a[NAMES.index(k)][:] = v
    else:
        a[NAMES.index('t')][:] = np.linspace(285., 220., a.shape[1])
        a[NAMES.index('qr')][:10] = 2e-4; a[NAMES.index('nr')][:10] = 3e3
        sp = 'qs' if label == 'melting_snow_top' else 'qg'; nsp = 'ns' if sp == 'qs' else 'ng'
        a[NAMES.index(sp)][:] = np.linspace(1e-5, 8e-4, a.shape[1]); a[NAMES.index(nsp)][:] = 2e4
    cols.append(a); provenance.append('synthetic:' + label)
arr = np.stack(cols).astype(np.float32)  # (ncol, 9, nz)
ncol, _, nz = arr.shape
np.savez_compressed(out / 'inputs.npz', columns=arr, provenance=np.asarray(provenance))
for dt, name in ((np.float32, 'input_fp32.bin'), (np.float64, 'input_fp64.bin')):
    with open(out / name, 'wb') as fh:
        fh.write(np.asarray([nz, ncol], np.int32).tobytes()); fh.write(arr.astype(dt).tobytes())
print(ncol, nz, hashlib.sha256((out / 'inputs.npz').read_bytes()).hexdigest()[:16], (out / 'inputs.npz').stat().st_size)
