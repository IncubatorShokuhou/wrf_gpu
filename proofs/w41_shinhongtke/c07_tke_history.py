"""Measured TKE_PBL / EL_PBL / QKE history of original CPU-WRF vs the port (bl=11, diag=0).

Reads existing Swiss c07 (v-prep, src 44b7b226b) and c07tw1 (w41-topowind, src 922e4de092)
CPU-WRF and GPU wrfout frames (not ALISIOS data). No JAX.
usage: python c07_tke_history.py <out.json>
"""
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import netCDF4

CASES = {
    'c07_vprep_44b7b226b': Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c07'),
    'c07tw1_topowind_922e4de092': Path('<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/cases/c07tw1'),
}
FRAMES = ('2023-01-15_00:00:00', '2023-01-15_01:00:00', '2023-01-15_12:00:00', '2023-01-16_00:00:00')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def levels(a):
    """Distinct values per level group: mass levels and Z top."""
    a = np.asarray(a, np.float32)
    return dict(shape=list(a.shape), mass_unique=[float(x) for x in np.unique(a[:-1])[:4]],
                top_unique=[float(x) for x in np.unique(a[-1])[:4]] if a.shape[0] == 45 else None,
                mass_hex=sorted({np.float32(x).tobytes().hex() for x in np.unique(a[:-1])[:4]}))


out = {}
for tag, root in CASES.items():
    rows = {}
    nml = (root / 'input' / 'namelist.input')
    diag = [l.strip() for l in nml.read_text().splitlines() if 'shinhong' in l.lower()] if nml.exists() else None
    bl = [l.strip() for l in nml.read_text().splitlines() if 'bl_pbl_physics' in l] if nml.exists() else None
    for frame in FRAMES:
        cpu = root / 'cpu' / f'wrfout_d01_{frame}'
        gpu = sorted(glob.glob(str(root / 'gpu' / '**' / f'wrfout_d01_{frame}'), recursive=True))
        if not cpu.exists() or not gpu:
            continue
        with netCDF4.Dataset(cpu) as c, netCDF4.Dataset(gpu[0]) as g:
            row = dict(cpu_sha256=sha(cpu), gpu_sha256=sha(gpu[0]))
            for name in ('TKE_PBL', 'EL_PBL', 'QKE'):
                row[name] = dict(cpu=levels(c[name][0]) if name in c.variables else 'absent',
                                 gpu=levels(g[name][0]) if name in g.variables else 'absent')
            rows[frame] = row
    out[tag] = dict(namelist_bl=bl, namelist_shinhong_lines=diag, frames=rows)
out['reading'] = ('[M] CPU-WRF TKE_PBL = REAL 0.005 (0x0ad7a33b) on all 44 mass levels from lead 0, 0 at the Z top; '
                  'GPU TKE_PBL = 0 everywhere (writer had no bl=11 source, generic zero fallback). GPU QKE (superset '
                  'variable, absent in CPU-WRF) shows the held carry: 0 at lead 0 (t0 overlay from wrfinput), 0.005 later. '
                  'EL_PBL 0 in both (diag=0). Fix: writer fills the TKE_PBL slot from State.qke/cam_tke_pbl.')
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: {f: {n: (v[n]['cpu'] if isinstance(v[n]['cpu'], str) else v[n]['cpu']['mass_unique'],
                              v[n]['gpu'] if isinstance(v[n]['gpu'], str) else v[n]['gpu']['mass_unique'])
                          for n in ('TKE_PBL', 'EL_PBL', 'QKE')} for f, v in c['frames'].items()}
                  for k, c in out.items() if k != 'reading'}, indent=1))
