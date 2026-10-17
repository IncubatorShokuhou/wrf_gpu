"""O1 deviceless selected-program/carry/constant identity, never fidelity.

Use the same CPU-safe Thomas substitution in both compared sources. Pallas
loads interpret on CPU; the traced production step uses its native facade.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

source, case, output = map(Path, sys.argv[1:4])
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0, str(source / 'src'))
sys.path.insert(0, str(source / 'proofs/o1_nlbind'))
import jax
import numpy as np
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from jax.experimental import pallas as pl

assert jax.devices()[0].platform == 'cpu'
assert Path(om.__file__).is_relative_to(source)
state_contract._gpu_device = lambda: jax.devices('cpu')[0]
original_pallas_call = pl.pallas_call


def interpret(*args, **kwargs):
    kwargs['interpret'] = True
    kwargs.pop('compiler_params', None)
    return original_pallas_call(*args, **kwargs)


def fingerprint(value):
    value = np.asarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


started = time.monotonic()
with tempfile.TemporaryDirectory(prefix='load-', dir=output.parent) as temporary:
    cfg = pipeline.NestedPipelineConfig(case, Path(temporary) / 'output',
        Path(temporary) / 'proof', hours=1, max_dom=1)
    pl.pallas_call = interpret
    try:
        _, bundles, _, _, _, carries = pipeline._load_domains(cfg, ('d01',))
    finally:
        pl.pallas_call = original_pallas_call
ns, carry = bundles['d01'].namelist, carries['d01']
om._resolve_operational_suite(ns)
clock = om.build_clock_base(ns)
initial = {str(path): fingerprint(value)
           for path, value in jax.tree_util.tree_leaves_with_path(carry)}
loaded = time.monotonic()
jax.clear_caches()
program = jax.make_jaxpr(lambda c: om._advance_chunk_fori(c, ns, 1, clock,
    n_steps=1, cadence=int(ns.radiation_cadence_steps)))(carry)
text = re.sub(r'name_and_src_info=[^\]\n]*', '', str(program))
text = re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?', '', text)
receipt = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
    cwd=source, text=True).strip(), 'case': str(case), 'platform': 'cpu',
    'GPU_executed': False, 'numerical_steps_executed': 0,
    'selected_canon_sha256': hashlib.sha256(text.encode()).hexdigest(),
    'canonical_chars': len(text), 'initial_carry': initial,
    'closed_constants': [fingerprint(v) for v in program.consts],
    'load_s': loaded - started, 'wall_s': time.monotonic() - started,
    'work': {k: getattr(ns, k) for k in ('dt_s', 'mp_physics', 'bl_pbl_physics',
        'sf_sfclay_physics', 'sf_surface_physics', 'cu_physics', 'ra_lw_physics',
        'ra_sw_physics', 'radiation_cadence_steps', 'cumulus_cadence_steps')},
    'scope': 'Accepted O1 selected native-facade trace with identical CPU-safe tridiagonal substitution; no CUDA executable realization, forecast or fidelity claim'}
output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items()
                  if k not in ('initial_carry', 'closed_constants')}, indent=2))
