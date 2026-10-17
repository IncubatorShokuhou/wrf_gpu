"""Freeze a tracked subset of the pristine-WRF surface-layer oracle (lane t-cfix-sfc).

usage: python make_fixture.py <lane_dir>
Reads <lane>/oracle/{oracle.npz,oracle.json,envelope.npz,manifest.json} and writes
proofs/t_cfix_sfc/fixtures/sfclay_wrf_oracle.{npz,json}: every second column (all
surface classes and regimes stay represented), both chained calls per scheme, the
zero-build outputs and WRF's own +-1 ULP input envelope (with the FMA build).
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np

lane = Path(sys.argv[1]) / 'oracle'
here = Path(__file__).resolve().parent
d = np.load(lane / 'oracle.npz'); env = np.load(lane / 'envelope.npz')
meta = json.loads((lane / 'oracle.json').read_text())
rows = np.arange(0, d['inputs'].shape[0], 2)
arrays = {'rows': rows, 'inputs': d['inputs'][rows]}
for s in (91, 7, 3):
    arrays[f'sf{s}_inp2'] = d[f'sf{s}_zero_inp2'][rows]
    for c in (1, 2):
        arrays[f'sf{s}_out{c}'] = d[f'sf{s}_zero_out{c}'][rows]
        arrays[f'sf{s}_lower{c}'] = env[f'sf{s}_lower{c}'][rows]
        arrays[f'sf{s}_upper{c}'] = env[f'sf{s}_upper{c}'][rows]
regimes = {s: sorted({int(r) for c in (1, 2) for r in arrays[f'sf{s}_out{c}'][:, meta['outputs'].index('regime')]})
           for s in (91, 7)}
out = here / 'fixtures'
out.mkdir(exist_ok=True)
path = out / 'sfclay_wrf_oracle.npz'
np.savez_compressed(path, **arrays)
labels = [meta['labels'][i] for i in rows]
fixture = dict(
    fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source='WRF 4.7.1 phys/module_sf_sfclay.F (91), module_sf_pxsfclay.F (7), module_sf_gfs.F (3); unmodified',
    build_manifest=meta['build_manifest'], oracle_npz_sha256=meta['npz_sha256'],
    envelope_draws=int(env['draws']), inputs=meta['inputs'], outputs=meta['outputs'],
    surfaces=sorted({lab['surface'] for lab in labels}), regimes=regimes, n_columns=int(rows.size),
    rule='per call on identical operands: port inside [lower, upper] of WRF zero/FMA builds over +-1 ULP '
         'input draws, +-2 REAL4 output ULPs; SFCLAY regime-1 ZOL (entry value kept by WRF, no consumer) excluded',
    columns='lowest level of ORIGINAL CPU-WRF c04 (Swiss 2023-01-15) + water/calm variants; '
            'call 2 consumes call-1 INOUT memory (UST ZNT MOL ZOL QSFC HFX QFX REGIME; SF_GFS UST ZNT)',
)
(out / 'sfclay_wrf_oracle.json').write_text(json.dumps(fixture, indent=1) + '\n')
print(fixture['fixture_sha256'], path.stat().st_size, rows.size, regimes, fixture['surfaces'])
