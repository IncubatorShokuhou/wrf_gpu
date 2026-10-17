"""Delete the real denominator guard; require a numerical assertion, zero errors."""
import contextlib
import inspect
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts/v025'))
import cpu_guard  # enforce backend before importing gpuwrf/JAX

import pytest

from gpuwrf.physics import urban_bep_components as k

assert os.environ['JAX_PLATFORMS'] == 'cpu'
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
source = inspect.getsource(k.bem_flux_flat)
original = 'p0 = jax.lax.optimization_barrier(jnp.asarray(1.e5, dtype=jnp.asarray(theta).dtype))'
replacement = 'p0 = jnp.asarray(1.e5, dtype=jnp.asarray(theta).dtype)'
assert source.count(original) == 1
exec(compile(source.replace(original,replacement), '<bep-denominator-mutant>', 'exec'), k.__dict__)
with (out/'mutant.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    rc = pytest.main(['-q','tests/test_v040_urban_bep_components.py::test_surface_fluxes[fp32-3]',
                      '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s1-bepbem/pytest',
                      '--junitxml='+str(out/'mutant.xml')])
xml = ET.parse(out/'mutant.xml')
failures = xml.findall('.//failure')
errors = xml.findall('.//error')
assert rc == 1 and len(failures) == 1 and not errors
assert 'AssertionError' in failures[0].get('message','')
assert 'Mismatched elements' in (failures[0].text or '')
(out/'mutant.json').write_text(json.dumps({'kill':'numerical AssertionError','failures':1,'errors':0,
                                         'mutant':'delete actual pressure-division guard'},indent=2)+'\n')
print('KILLED: 1 numerical assertion, 0 setup errors')
