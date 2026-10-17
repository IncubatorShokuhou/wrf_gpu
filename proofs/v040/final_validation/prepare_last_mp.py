"""Matched original-WRF references for newly CLI-admitted NTU56/CAM-MG11."""
from pathlib import Path
import json
import os
import shutil
import sys
from prepare import ROOT, RETAINED, SLOTS, write_json, sha
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.scheme_catalog import _IMPLEMENTED

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3
source = Path(sys.argv[1]).resolve()
matrix = json.loads((ROOT / 'MATRIX.json').read_text())
for ident, mp, pbl in (('c88', 56, 5), ('c89', 11, 9)):
    assert mp in _IMPLEMENTED['mp_physics'], 'CLI admission must precede staging'
    assert not any(r['id'] == ident for r in matrix['cases'])
    case = ROOT / ident
    data = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms') / ident / 'cpu'
    data.mkdir(parents=True, exist_ok=False)
    case.mkdir(exist_ok=False)
    (case / 'cpu').symlink_to(data, target_is_directory=True)
    groups = parse_namelist(source / 'examples/switzerland_d01/namelist.input')
    groups['physics'].update(mp_physics=mp, cu_physics=0, bl_pbl_physics=pbl,
                             sf_sfclay_physics=5, use_mp_re=1)
    row = dict(id=ident, geography='Swiss', hours=24, domains=['d01'], extra={},
        options={k: groups['physics'][k] for k in SLOTS},
        features=[f'primary-MP{mp}-admitted-batch15', 'original-WRF-inputs-no-synthetic-fields'])
    row = stage(row, RETAINED / 'source_inputs/Swiss', groups)
    assert not row['admission_errors'], row['admission_errors']
    # Read-only forcing links are allowed; mutable controls are regular OWNED
    # bytes. Never copy then write a foreign CPU/input control symlink (E165).
    control = case / 'cpu/namelist.input'
    assert control.is_symlink()
    control.unlink()
    control.write_bytes((case / 'input/namelist.input').read_bytes())
    assert not control.is_symlink() and control.resolve().is_relative_to(data)
    assert sha(control) == row['input_sha256']['namelist.input']
    for p in list(data.iterdir()):
        if p.is_symlink() and (p.name == 'namelist.output' or p.name == 'tslist'
                or p.name.startswith(('rsl.', 'wrfout_', 'wrfrst_'))):
            p.unlink()  # The owned link only, never its foreign target.
    row['reference_semantics'] = 'Original pristine serial52f executable, two reversible diagnostic OPEN patches only; no zero-init substitution unless required by MYNN policy'
    write_json(case / 'inputs.json', row)
    matrix['cases'].append(row)
    write_json(ROOT / 'MATRIX.json', matrix)
    print(ident, 'ADMITTED', row['options'], flush=True)
