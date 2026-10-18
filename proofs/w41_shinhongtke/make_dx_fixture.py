"""Freeze the dx=1 km / 9 km pristine-WRF Shin-Hong rows (lane w41-shinhongtke) as a tracked fixture.

usage: python make_dx_fixture.py <dx_root>   (dx_root/dx1000, dx_root/dx9000 from build_dx_oracle.py
+ ulp_envelope.py). Rows: the 8 real c07 columns of proofs/t_shinhong/fixtures (stable pblflg=F
kpbl>1, cloud water/ice, convective entrainment) in modes 0, 1 cold, 1 warm, at both dx -> 48 rows.
Layout identical to proofs/t_shinhong/fixtures/shinhong_wrf_oracle.npz (+ dx per row).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]); here = Path(__file__).resolve().parent
warm = [38, 44, 50, 62, 68, 71, 74, 95]
keys = ('profiles', 'surface', 'nz', 'mode', 'wrf', 'lower', 'upper', 'has_envelope', 'dx', 'oracle_row')
parts = {k: [] for k in keys}; meta_rows, sources = [], {}
for dx in (1000, 9000):
    lane = root / f'dx{dx}'
    d = np.load(lane / 'oracle.npz'); env = np.load(lane / 'ulp_envelope.npz')
    manifest = json.loads((lane / 'oracle_manifest.json').read_text())
    rows = sorted(i for w in warm for i in (w - 2, w - 1, w))
    env_rows = list(env['rows'])
    for pos, i in enumerate(rows):
        r = manifest['rows'][i]
        assert r['label'] == ('mode0', 'mode1_cold', 'mode1_warm')[pos % 3] and r['source'] == 'c07' and r['dx'] == dx
        lo = env['lower'][i] if i in env_rows else d['out_zero'][i]
        hi = env['upper'][i] if i in env_rows else d['out_zero'][i]
        for k, v in (('profiles', d['profiles'][i]), ('surface', d['surface'][i]), ('nz', d['nz'][i]), ('mode', d['mode'][i]),
                     ('wrf', d['out_zero'][i]), ('lower', lo), ('upper', hi), ('has_envelope', i in env_rows),
                     ('dx', dx), ('oracle_row', i)):
            parts[k].append(v)
        meta_rows.append(dict(r, oracle_row=i))
    sources[dx] = dict(oracle_npz_sha256=manifest['oracle_npz_sha256'],
                       envelope=json.loads((lane / 'ulp_envelope.json').read_text()),
                       init_sensitivity_zero_vs_inf_byte=manifest['init_sensitivity_byte_identical'])
out = here / 'fixtures'; out.mkdir(exist_ok=True)
path = out / 'shinhong_dx_oracle.npz'
np.savez_compressed(path, **{k: np.asarray(v) for k, v in parts.items()})
first = json.loads((root / 'dx1000' / 'oracle_manifest.json').read_text())
meta = dict(fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source='WRF 4.7.1 phys/module_bl_shinhong.F (pristine, t-shinhong lane builds reused by SHA)',
            source_sha256=first['source_sha256'], wrapper_sha256=first['wrapper_sha256'],
            reference_build=first['builds']['zero'], fma_build=first['builds']['fma'], per_dx=sources, rows=meta_rows,
            rule='literal path BYTE (subnormal-range qc/qi excluded, E117); compiled path inside [lower, upper] +- 2 REAL4 output ULPs, no exceptions')
(out / 'shinhong_dx_oracle.json').write_text(json.dumps(meta, indent=1) + '\n')
print(meta['fixture_sha256'], path.stat().st_size, len(meta_rows))
