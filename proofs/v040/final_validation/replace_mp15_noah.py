"""Move the excluded MP15 row's independent Noah options to a valid MP8 row."""
import json
import os
from pathlib import Path
from prepare import ROOT, RETAINED, SLOTS, private_executable, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix = json.loads((ROOT / 'MATRIX.json').read_text())
assert not any(r['id'] == 'c86' for r in matrix['cases'])
groups = parse_namelist(ROOT / 'c63/input/namelist.input')
groups['physics']['mp_physics'] = 8
row = dict(id='c86', geography='Swiss', hours=24, domains=['d01'], extra={},
    options={k: groups['physics'][k] for k in SLOTS},
    features=['Noah-options-repacked-from-owner-excluded-MP15',
              'dveg2/run2/btr3/rad2/snf3/stc3/rsf3'])
row = stage(row, RETAINED / 'source_inputs/Swiss', groups)
assert not row['admission_errors'], row['admission_errors']
private_executable(ROOT / 'c86', ROOT / 'reference_zero')
row['reference_semantics'] = 'compiler-defined zero, inactive canonical Morrison-aero object'
write_json(ROOT / 'c86/inputs.json', row)
matrix['cases'].append(row)
write_json(ROOT / 'MATRIX.json', matrix)
print(json.dumps(row, indent=2))
