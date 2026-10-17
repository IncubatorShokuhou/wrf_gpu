"""Read-only original-WRF frame-zero reconstruction; no model forecast."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path('<USER_HOME>/wrf_gpu2_lanes')
sources = {
    'input': BASE/'s1-gpuqual/source_inputs/Swiss/wrfinput_d01',
    'mp28_cpu': BASE/'s1-gpuqual/mp28/cpu/wrfout_d01_2023-01-15_00:00:00',
    'c19_cpu': BASE/'v-prep/c19/cpu/wrfout_d01_2023-01-15_00:00:00',
    'old_gpu': BASE/'v-run/round3/mp28_followup/mp28/gpu/wrfout_d01_2023-01-15_00:00:00',
}
arrays = {}
report = {'sources': {}, 'reconstructions': {}}
for label, path in sources.items():
    report['sources'][label] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        names = ('PH', 'PHB', 'QNWFA', 'QNIFA', 'QNWFA2D', 'QNIFA2D', 'HGT')
        report['sources'][label]['fields'] = {}
        for name in names:
            if name not in ds.variables:
                continue
            a = np.asarray(ds[name][0])
            if name in ('PH', 'PHB', 'QNWFA', 'QNIFA'):
                a = a[:2]
            arrays[label+'_'+name] = a
            report['sources'][label]['fields'][name] = {'min': float(a.min()), 'max': float(a.max()), 'dtype': str(a.dtype)}

ref = arrays['mp28_cpu_QNWFA2D']
def comparison(a):
    diff = np.abs(a.astype('f8')-ref)
    ulp = np.abs(a.astype('f4').view('i4').astype('i8')-ref.view('i4'))
    return {'mean': float(a.mean()), 'max_abs': float(diff.max()), 'max_ulp': int(ulp.max()),
            'exact_cells': int(np.count_nonzero(a == ref)), 'cells': a.size,
            'j21_i18': float(a[21,18])}

report['c19_vs_mp28'] = comparison(arrays['c19_cpu_QNWFA2D'])
for geometry in ('interface', 'mass'):
    for dtype in ('f4', 'f8'):
        scalar = np.dtype(dtype).type
        # WRF start_em.F:873: z_at_q=(ph_2+phb)/g, with REAL4 g=9.81.
        with Dataset(sources['input']) as ds:
            ph = np.asarray(ds['PH'][0,:3], dtype=dtype)
            phb = np.asarray(ds['PHB'][0,:3], dtype=dtype)
        z = (ph+phb)/scalar(9.81)
        if geometry == 'mass':
            z = scalar(.5)*(z[:-1]+z[1:])
        h0 = z[0]
        h01 = np.where(h0 <= scalar(1000), scalar(.8), np.where(h0 >= scalar(2500), scalar(.01), scalar(.8)*np.cos(h0*scalar(.001)-scalar(1))))
        ni = -np.log(scalar(50e6)/scalar(300e6))/h01
        dz = z[1]-h0
        nwfa = scalar(50e6)+scalar(300e6)*np.exp(-(dz/scalar(1000))*ni)
        emission = nwfa*scalar(.000196)*(scalar(50)/dz)
        report['reconstructions'][geometry+'_'+dtype] = comparison(emission)
        if geometry == 'interface' and dtype == 'f4':
            arrays['oracle_interface_heights'] = z[:2]
            arrays['oracle_numpy_emission'] = emission
report['old_gpu'] = comparison(arrays['old_gpu_QNWFA2D'])
np.savez_compressed(OUT/'original_frame0.npz', **arrays)
(OUT/'TRACE.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, indent=2))
