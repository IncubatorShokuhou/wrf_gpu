"""CPU byte/program gate using the actual old allocation/radiation functions.

Usage: gate.py SOURCE BASELINE PRE_CAM_COMMIT OUTPUT
No model/GPU compile. Old functions execute in the same imported environment,
isolating this patch from unrelated source changes and cold-compile choices.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

root, baseline, pre_cam, output = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'src'))
import jax
import numpy as np
from gpuwrf.contracts import state as S
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_mp_re as R

assert jax.devices()[0].platform == 'cpu'
helpers = runpy.run_path(str(Path(root) / 'tests/v040/test_cammg_activation.py'))
state, scan, byte_equal = (helpers[n] for n in ('_state', '_scan', '_byte_equal'))


def function(source, name, globals_):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
    return namespace[name]


def canon(program):
    text = re.sub(r'name_and_src_info=[^\]\n]*', '', str(program))
    return re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?', '', text)


old_state_source = (Path(baseline) / 'src/gpuwrf/contracts/state.py').read_text()
old_radiation_source = (Path(baseline) / 'src/gpuwrf/coupling/physics_couplers.py').read_text()
pre_source = subprocess.check_output(['git', '-C', root, 'show',
    pre_cam + ':src/gpuwrf/coupling/physics_couplers.py'], text=True)
old_selector = function(old_state_source, 'conditional_state_leaves_for_mp', vars(S))
new_selector = S.conditional_state_leaves_for_mp
old_rad = function(old_radiation_source, '_rrtmg_column_inputs_impl', vars(C))
pre_rad = function(pre_source, '_rrtmg_column_inputs_impl', vars(C))
new_rad = C._rrtmg_column_inputs_impl
R._MP_RE = True
start = time.monotonic()
result = {'platform': 'cpu', 'source': root, 'baseline': baseline,
          'pre_cam_commit': pre_cam, 'cache': 'OFF', 'arms': {}}
for mp in (6, 8, 11, 24, 28, 40, 50, 55, 56):
    original = state(mp)
    try:
        S.conditional_state_leaves_for_mp = old_selector
        before = scan(original, mp)
    finally:
        S.conditional_state_leaves_for_mp = new_selector
    after = scan(original, mp)
    for name in after.active_field_names():
        byte_equal(getattr(before, name), getattr(after, name))
    removed = set(before.active_field_names()) - set(after.active_field_names())
    assert removed == (set() if mp == 11 else set(S.CAMMG_STATE_FIELDS))
    rec = {'common_state_BYTE': True, 'removed_fields': sorted(removed),
           'removed_bytes_small_grid': sum(getattr(before, n).nbytes for n in removed)}
    kwargs = dict(time_utc='2026-02-28_12:00:00', lead_seconds=18.,
                  mp_physics=mp, cloud_mp_physics=mp, use_mynn_sgs_cloud=False)
    reference = old_rad if mp == 11 else pre_rad
    reference_state = before if mp == 11 else after
    # Compiled equality (not an eager-only claim) and the entire traced program.
    expected = jax.jit(lambda s: reference(s, None, **kwargs))(reference_state)
    actual = jax.jit(lambda s: new_rad(s, None, **kwargs))(after)
    byte_equal(actual, expected)
    old_program = jax.make_jaxpr(lambda s: reference(s, None, **kwargs))(reference_state)
    new_program = jax.make_jaxpr(lambda s: new_rad(s, None, **kwargs))(after)
    assert canon(old_program) == canon(new_program), f'program mp{mp}'
    byte_equal(old_program.consts, new_program.consts)
    rec.update(radiation_compiled_BYTE=True, radiation_program_EXACT=True,
               constants_BYTE=True, program_sha256=hashlib.sha256(canon(new_program).encode()).hexdigest())
    if mp == 8:
        broken = old_rad(before, None, **kwargs)
        rec['old_public_entry_cloud_max'] = float(np.asarray(broken[0].cloud_fraction).max())
        rec['fixed_public_entry_cloud_max'] = float(np.asarray(actual[0].cloud_fraction).max())
        rec['old_radius_microns'] = float(np.asarray(broken[0].re_cloud).max())
        rec['fixed_radius_meters'] = float(np.asarray(actual[0].re_cloud).max())
        assert rec['old_public_entry_cloud_max'] == 0
        assert rec['fixed_public_entry_cloud_max'] > 0
        assert not np.array_equal(np.asarray(broken[0].re_cloud), np.asarray(actual[0].re_cloud))
    result['arms'][str(mp)] = rec
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
result['elapsed_s'] = time.monotonic() - start
result['status'] = 'PASS'
Path(output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
