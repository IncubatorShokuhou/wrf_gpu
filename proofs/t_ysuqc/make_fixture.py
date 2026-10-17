"""Freeze a tracked subset of the pristine-WRF YSU oracle (lane t-ysuqc).

usage: python make_fixture.py <lane_dir>
Reads <lane>/oracle.npz, oracle_manifest.json (build_oracle.py) and
ulp_envelope.npz/.json (ulp_envelope.py); writes
proofs/t_ysuqc/fixtures/ysu_wrf_oracle.npz + .json. Columns (both
ysu_topdown_pblmix modes each): every column whose WRF result depends on the
top-down switch, cloud-ice columns (nonzero RQIBLTEN), the FLAG_QI=.false.
Kessler columns with cloud water, and clear/stable/water controls.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np

lane = Path(sys.argv[1])
here = Path(__file__).resolve().parent
d = np.load(lane / 'oracle.npz'); env = np.load(lane / 'ulp_envelope.npz')
manifest = json.loads((lane / 'oracle_manifest.json').read_text())
rows_meta = manifest['rows']
z = d['out_zero']
pairs = list(range(0, len(rows_meta), 2))
assert all(rows_meta[a]['topdown'] == 1 and rows_meta[a + 1]['topdown'] == 0 for a in pairs)
td = [a for a in pairs if np.any(z[a] != z[a + 1])]
ice = [a for a in pairs if rows_meta[a]['source'] == 'c07' and rows_meta[a]['pick'] in ('max_qi_col', 'max_qi_low')
       and np.abs(z[a, :-1, 5]).max() > 0]
kessler = [a for a in pairs if rows_meta[a]['source'] == 'c02' and rows_meta[a]['pick'] in ('min_hfx', 'random1')
           and rows_meta[a]['frame'].endswith('06:00:00')]
controls = [a for a in pairs if (rows_meta[a]['source'], rows_meta[a]['frame'][11:13], rows_meta[a]['pick'])
            in {('c40', '12', 'max_hfx'), ('c40', '00', 'min_hfx'), ('c85', '06', 'max_cloud_pbltop'),
                ('c85', '12', 'max_qc_low')}]
cols = sorted(set(td + ice + kessler + controls))
rows = [r for a in cols for r in (a, a + 1)]
out = here / 'fixtures'
out.mkdir(exist_ok=True)
path = out / 'ysu_wrf_oracle.npz'
np.savez_compressed(path, profiles=d['profiles'][rows], surface=d['surface'][rows], nz=d['nz'][rows],
                    topdown=d['topdown'][rows], flag_qi=d['flag_qi'][rows], wrf=z[rows],
                    lower=env['lower'][rows], upper=env['upper'][rows])
meta = dict(
    fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source='WRF 4.7.1 phys/module_bl_ysu.F + phys/physics_mmm/bl_ysu.F90 (pristine, unmodified lane copies)',
    source_sha256=manifest['source_sha256'], wrapper='proofs/t_ysuqc/ysu_oracle_wrapper.f90',
    wrapper_sha256=manifest['wrapper_sha256'], reference_build=manifest['builds']['zero'],
    fma_build=manifest['builds']['fma'], oracle_npz_sha256=manifest['oracle_npz_sha256'],
    init_zero_vs_inf_consumed_outputs_byte=True,
    init_note='RQIBLTEN is never assigned when FLAG_QI=.false. (bl_ysu.F90:1205 qitnp INTENT(OUT)); '
              'WRF does not consume it then; every other output zero==inf BYTE',
    envelope=json.loads((lane / 'ulp_envelope.json').read_text()),
    census=dict(columns=len(cols), topdown_sensitive=len(td), cloud_ice=len(ice), kessler_flag_qi_false=len(kessler),
                controls=len(controls)),
    rows=[dict(rows_meta[i], oracle_row=i) for i in rows],
    layout=dict(profiles='u v t qv qc qi p p8w(n+1) pi dz8w rthraten',
                surface='psfc znt ust psim(fm) psih(fhh) xland hfx qfx wspd br u10 v10 uoce voce ctopo ctopo2 dt',
                wrf='rublten rvblten rthblten rqvblten rqcblten rqiblten exch_h exch_m scalars(hpbl,kpbl,wstar,delta,u10,v10)'),
    rule='literal path BYTE with exp/tanh through glibc (q rates below 1e-30 absolute, E117); '
         'compiled path inside WRF +-1 ULP input envelope widened by 2 REAL4 output ULPs',
)
(out / 'ysu_wrf_oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
print(meta['fixture_sha256'], path.stat().st_size, len(rows), meta['census'])
