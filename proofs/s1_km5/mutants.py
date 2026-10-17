"""Genuine numeric mutant; run ONE per fresh CPU process, no checkout edits."""
from pathlib import Path
import inspect
import sys

ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v025'))
import cpu_guard  # Must precede JAX imports, including when pytest runs in-process.
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(ROOT / 'src'))
from gpuwrf.dynamics import sms3dtke as sms

name = sys.argv[1]
key = 'grey_spec'
if name == 'drop_scale_blends':
    sms._partial = lambda d, h, *, local: jnp.zeros_like(d)
elif name == 'drop_implicit_dissipation_length':
    sms._sms_l_diss = lambda m, cfg, tke, *a: jnp.zeros_like(tke)
elif name == 'wrong_seed_epsilon':
    src = inspect.getsource(sms._sms_km).replace('1.e-10', '1.e-15')
    assert src != inspect.getsource(sms._sms_km)
    namespace = vars(sms).copy()
    exec(compile(src, '<seed-epsilon-mutant>', 'exec'), namespace)
    sms._sms_km = namespace['_sms_km']
    key = 'drag_seed_edge'
else:
    raise ValueError(name)

rc = pytest.main(['-q', str(ROOT / 'tests/dynamics/test_sms3dtke.py') +
    f'::test_sms_components_match_pristine[False-{key}]',
    '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/s1-km5/pytest'])
# Caller also requires AssertionError/numeric mismatch and zero setup errors.
assert rc == 1, (name, rc)
print('EXPECTED_NUMERIC_FAILURE', name)
