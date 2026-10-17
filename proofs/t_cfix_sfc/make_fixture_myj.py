"""Freeze a tracked subset of the pristine-WRF MYJSFC oracle (lane t-cfix-sfc).

usage: python make_fixture_myj.py <lane_dir>  -> proofs/t_cfix_sfc/fixtures/myjsfc_wrf_oracle.{npz,json}
Every second column; three chained calls (itimestep 1 cold from Registry-zero memory,
2 and 3 consuming the previous call's INOUT memory); zero-build outputs and WRF's own
+-1 ULP input envelope (incl. the FMA build).
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np

lane = Path(sys.argv[1]) / 'oracle'
here = Path(__file__).resolve().parent
d = np.load(lane / 'oracle_myj.npz'); env = np.load(lane / 'envelope_myj.npz')
meta = json.loads((lane / 'oracle_myj.json').read_text())
rows = np.arange(0, d['prof'].shape[0], 2)
arrays = {'rows': rows, 'prof': d['prof'][rows]}
for c in (1, 2, 3):
    arrays[f'sfc{c}'] = d[f'zero_sfc{c}'][rows]
    arrays[f'out{c}'] = d[f'zero_out{c}'][rows]
    arrays[f'lower{c}'] = env[f'lower{c}'][rows]
    arrays[f'upper{c}'] = env[f'upper{c}'][rows]
path = here / 'fixtures' / 'myjsfc_wrf_oracle.npz'
np.savez_compressed(path, **arrays)
labels = [meta['labels'][i] for i in rows]
manifest = json.loads((lane / 'manifest.json').read_text())
fixture = dict(
    fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source='WRF 4.7.1 phys/module_sf_myjsfc.F + share/module_model_constants.F (unmodified)',
    sources={k: v for k, v in manifest['sources'].items() if k in ('module_sf_myjsfc.F', 'module_model_constants.F')},
    builds={k: v for k, v in manifest['builds'].items() if k.startswith('sf2/')},
    wrapper_sha256=manifest['wrappers'].get('myjsfc_oracle.f90'), oracle_npz_sha256=meta['npz_sha256'],
    envelope_draws=int(env['draws']), prof=meta['prof'], sfc=meta['sfc'], outputs=meta['outputs'],
    surfaces=sorted({lab['surface'] for lab in labels}), n_columns=int(rows.size),
    rule='per call on identical operands: port inside [lower, upper] (+-1 ULP input draws of the zero build '
         'and the FMA build) +-8 REAL4 output ULPs (compiled REAL path; 123 of 79380 cells need 3-8 ULP: '
         '5-iteration SFCDIF fixed point, XLA vs glibc libm; disclosed)',
    columns='ORIGINAL CPU-WRF c03 (Swiss 2023-01-15, bl=2/sf=2/Noah-MP) 44-level profiles + water/calm variants',
)
(here / 'fixtures' / 'myjsfc_wrf_oracle.json').write_text(json.dumps(fixture, indent=1) + '\n')
print(fixture['fixture_sha256'], path.stat().st_size, rows.size, fixture['surfaces'])
