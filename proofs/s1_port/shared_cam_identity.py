"""Reuse the qualified CAM shared smoke only when its program/constants agree.

Adapted from u2-diag's shared_trace.py (UD24/27); no numerical re-execution.
Usage: python shared_cam_identity.py IMMUTABLE_TREE OUTPUT_JSON
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root, output = (Path(p).resolve() for p in sys.argv[1:3])
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip()
sys.path[:0] = [str(root / 'src'), str(root / 'tests')]
import jax
import numpy as np
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
import test_v013_operational_smoke as smoke

assert jax.devices()[0].platform == 'cpu'
assert Path(om.__file__).resolve().is_relative_to(root / 'src')
assert os.environ.get('GPUWRF_FAST_DEFAULTS') == '0'
reference = json.loads((Path(__file__).parent / 'receipts/CAM_REFERENCE.json').read_text())
grid = smoke._grid()
state = smoke._base_state(grid)
nl = smoke._namelist(grid, mp_physics=0, bl_pbl_physics=0,
                     sf_sfclay_physics=0, cu_physics=0, ra_sw_physics=3)
smoke._resolve_operational_suite(nl)
carry = initial_operational_carry(state)
start = time.perf_counter()
program = jax.make_jaxpr(lambda c: om._physics_step_forcing(
    c, nl, 0., run_radiation=True).carry.rthraten)(carry)
text = re.sub(r'name_and_src_info=[^\]\n]*', '', str(program))
text = re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?', '', text)
constants = []
for value in jax.tree.leaves(program.consts):
    array = np.asarray(value)
    constants.append(dict(dtype=array.dtype.str, shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest()))
sha = hashlib.sha256(text.encode()).hexdigest()
result = dict(source_commit=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    backend='cpu', gpu_executed=False, jaxpr_sha256=sha,
    captured_constants=len(constants), trace_s=time.perf_counter()-start,
    program_identical=sha == reference['case']['jaxpr_sha256'],
    all_constants_identical=constants == reference['case']['constants'],
    reference=reference['qualification'],
    scope='Reuse existing functional PASS; no native binary or new fidelity claim')
output.write_text(json.dumps(result, indent=2)+'\n')
assert result['program_identical'] and result['all_constants_identical'], result
print(json.dumps(result))
