"""Two actual MP_SCAN_ADAPTERS caller steps: original vs output-only firewall.

This is regression/neutrality evidence, not an independent WRF fidelity gate.
The retained original-CPU-WRF D6 receipts remain the fidelity evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('scheme', type=int, choices=(24, 26))
p.add_argument('out', type=Path)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0, str(a.source / 'src'))
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.physics import microphysics_wsm7 as wsm, microphysics_wdm7 as wdm
from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS

assert jax.devices()[0].platform == 'cpu'
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=a.source)
a.out.mkdir(parents=True, exist_ok=False)
config = NestedPipelineConfig(Path(f'<USER_HOME>/wrf_gpu2_lanes/v-run/mp{a.scheme}/input'),
                              a.out/'stream', a.out/'proof', hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(config, ('d01',))
initial = jax.tree.map(lambda v: np.asarray(v).copy(), carries['d01'].state)
nml = bundles['d01'].namelist
assert nml.mp_physics == a.scheme and nml.dt_s == 18
record = {'source': subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip(),
          'scheme': a.scheme, 'dt_s': nml.dt_s, 'platform': 'cpu', 'cache': 'NONE',
          'scope': __doc__, 'variant': 'output-only barrier on public MP run dict', 'arms': {},
          'caller': 'MP_SCAN_ADAPTERS[scheme], including production fp64 island and State replacements',
          'dynamics_and_other_physics': False, 'donation': False}

def run(label):
    jax.clear_caches()
    # Independent arrays per arm and owned output copies prevent donation/alias
    # reuse from turning a harness failure into a claimed numerical verdict.
    state = jax.tree.map(lambda v: jax.numpy.array(v, copy=True), initial)
    call = jax.jit(lambda s: MP_SCAN_ADAPTERS[a.scheme](s, float(nml.dt_s)))
    outputs = []
    rows = []
    for step in (1, 2):
        if Path('/tmp/wrf_gpu2_quiet').exists():
            raise RuntimeError('QUIET: refuse next heavy step')
        start = time.monotonic()
        state = call(state)
        jax.block_until_ready(state)
        leaves = [(jax.tree_util.keystr(k), np.asarray(v).copy())
                  for k,v in jax.tree_util.tree_leaves_with_path(state)]
        bad = [k for k,v in leaves if v.dtype.kind == 'f' and not np.isfinite(v).all()]
        assert not bad, (label, step, bad)
        outputs.append(leaves)
        rows.append({'step':step,'wall_s':time.monotonic()-start,'leaves':len(leaves),
                     'sha256':hashlib.sha256(b''.join(v.tobytes() for _,v in leaves)).hexdigest()})
        record['arms'][label] = rows
        (a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        print(label, rows[-1],flush=True)
    return outputs

base = run('original')
module, name = (wsm,'wsm7_run') if a.scheme == 24 else (wdm,'wdm7_run')
original = getattr(module, name)
def firewall(*args, **kwargs):
    return jax.lax.optimization_barrier(original(*args, **kwargs))
setattr(module, name, firewall)
candidate = run('firewall')
diffs = []
for step,(old,new) in enumerate(zip(base,candidate),1):
    assert len(old)==len(new)
    for (key,x),(key2,y) in zip(old,new):
        assert key==key2 and x.shape==y.shape and x.dtype==y.dtype
        if x.tobytes()!=y.tobytes():
            diffs.append({'step':step,'leaf':key,'max_abs':float(np.max(np.abs(x-y)))})
record.update(verdict='FAIL' if diffs else 'PASS', differing_leaves=diffs,
              byte_equal_leaves=sum(len(v) for v in base)-len(diffs))
(a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['verdict'],record['byte_equal_leaves'],'equal,',len(diffs),'different',flush=True)
raise SystemExit(bool(diffs))
