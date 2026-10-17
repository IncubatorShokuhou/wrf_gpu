"""Freeze a small tracked subset of the pristine-WRF Shin-Hong oracle (lane t-shinhong).

usage: python make_fixture.py <lane_dir>
Reads <lane>/oracle.npz + oracle_manifest.json (build_oracle.py) and
<lane>/ulp_envelope.npz (ulp_envelope.py); writes
proofs/t_shinhong/fixtures/shinhong_wrf_oracle.npz and its manifest JSON.
Columns: the 4 synthetic regimes and 8 real c07 columns that exercise the
stable pblflg=F kpbl>1 gate, cloud water/ice (in-cloud Ri, RQC/RQI, mixlen
cwm), and convective entrainment; every column in modes 0, 1 cold, 1 warm.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np

lane = Path(sys.argv[1])
here = Path(__file__).resolve().parent
d = np.load(lane / 'oracle.npz'); env = np.load(lane / 'ulp_envelope.npz')
manifest = json.loads((lane / 'oracle_manifest.json').read_text())
warm = [2, 5, 8, 11, 38, 44, 50, 62, 68, 71, 74, 95]   # mode1_warm row of each column
rows = sorted(i for w in warm for i in (w - 2, w - 1, w))
for i in rows:
    assert manifest['rows'][i]['label'] == ('mode0', 'mode1_cold', 'mode1_warm')[rows.index(i) % 3]
lower = np.array(d['out_zero'][rows]); upper = np.array(d['out_zero'][rows])
env_rows = list(env['rows'])
for pos, i in enumerate(rows):
    if i in env_rows:
        lower[pos] = env['lower'][env_rows.index(i)] if env['lower'].shape[0] == len(env_rows) else env['lower'][i]
        upper[pos] = env['upper'][env_rows.index(i)] if env['upper'].shape[0] == len(env_rows) else env['upper'][i]
out = here / 'fixtures'
out.mkdir(exist_ok=True)
path = out / 'shinhong_wrf_oracle.npz'
np.savez_compressed(path, profiles=d['profiles'][rows], surface=d['surface'][rows], nz=d['nz'][rows],
                    mode=d['mode'][rows], wrf=d['out_zero'][rows], lower=lower, upper=upper,
                    has_envelope=np.array([i in env_rows for i in rows]))
meta = dict(
    fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source='WRF 4.7.1 phys/module_bl_shinhong.F (pristine, unmodified lane copy)',
    source_sha256=manifest['source_sha256'], wrapper='proofs/t_shinhong/shinhong_oracle_wrapper.f90',
    wrapper_sha256=manifest['wrapper_sha256'], reference_build=manifest['builds']['zero'],
    fma_build=manifest['builds']['fma'], init_sensitivity_zero_vs_inf_byte=manifest['init_sensitivity_byte_identical'],
    oracle_npz_sha256=manifest['oracle_npz_sha256'],
    envelope=json.loads((lane / 'ulp_envelope.json').read_text()),
    rows=[dict(manifest['rows'][i], oracle_row=i) for i in rows],
    layout=dict(profiles='u v t qv p p8w(n+1) pi dz8w tke qc qi', surface='psfc znt ust hfx qfx wspd br fm fh dt xland u10 v10 dx dy corf ctopo ctopo2 hpbl_in',
                wrf='tke el exch_h rublten rvblten rthblten rqvblten rqcblten rqiblten scalars(hpbl,kpbl,wstar,delta,u10,v10,regime)'),
    rule='literal path BYTE (subnormal-range qc/qi excluded, E117); compiled path inside [lower, upper] +- 2 REAL4 output ULPs',
)
(out / 'shinhong_wrf_oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
print(meta['fixture_sha256'], path.stat().st_size, len(rows))
