"""Real Swiss CLI CPU restart gate; only State's device guard is adapted.

Environment: TRF_ROOT, TRF_ACTION=control|cut|resume. Every arm uses the
unmodified CPU physics/solver path. The cut arm exports the actual step-100
carry in all three standalone formats, then SIGKILLs itself after publication.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
import gpuwrf
import jax
import numpy as np
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import restart_store as storage

assert jax.devices()[0].platform == 'cpu'
state_contract._gpu_device = lambda: jax.devices('cpu')[0]
root = Path(os.environ['TRF_ROOT'])
action = os.environ['TRF_ACTION']
started = time.perf_counter()
bundles = {}


def hashes(tree):
    leaves, definition = jax.tree.flatten(tree)
    records = []
    for leaf in leaves:
        value = np.asarray(leaf)
        if value.dtype.kind in 'fc':
            assert np.isfinite(value).all()
        records.append([value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()])
    return {'treedef': str(definition), 'leaves': records}


load = pipeline._load_domains
def loaded(*args, **kwargs):
    result = load(*args, **kwargs)
    bundles.update(result[1])
    return result
pipeline._load_domains = loaded

advance = pipeline.run_operational_domain_tree
def advanced(*args, **kwargs):
    result = advance(*args, **kwargs)
    jax.block_until_ready(result.carries)
    record = {'action': action, 'platform': jax.devices()[0].platform,
              'source': str(Path(storage.__file__).resolve()),
              'own_steps': dict(result.own_steps), 'hashes': hashes(result.carries),
              'elapsed_s': time.perf_counter() - started}
    (root / (action + '_carry.json')).write_text(json.dumps(record, indent=1) + '\n')
    return result
pipeline.run_operational_domain_tree = advanced

save = storage.RestartStore.save
def saved(self, carries, own_steps, dt_s, identity, **kwargs):
    result = save(self, carries, own_steps, dt_s, identity, **kwargs)
    if action == 'cut' and own_steps['d01'] == 100:
        from gpuwrf.io.restart import write_restart
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry
        from gpuwrf.runtime.checkpoint import write_checkpoint
        carry, ns = carries['d01'], bundles['d01'].namelist
        formats = root / 'formats'
        formats.mkdir()
        write_restart(carry, ns, ns.grid, 100, formats / 'restart.pkl')
        write_checkpoint(carry.state, ns, ns.grid, 100, formats / 'checkpoint.pkl',
                         runtime_state=carry, land_state=carry.noahmp_land)
        write_wrfrst_carry(carry, ns.grid, ns, formats / 'wrfrst.nc',
                          run_start=identity['run_start'], valid_time='2023-01-15_00:30:00', step_index=100)
        (root / 'cut_generation.json').write_text(json.dumps({
            'generation': str(result[0]), 'receipt': result[1],
            'identity': identity, 'hashes': hashes(carries),
            'elapsed_s': time.perf_counter() - started}, indent=1) + '\n')
        print('TRF_SIGKILL_AFTER_VERIFIED', result[0], flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    return result
storage.RestartStore.save = saved

from gpuwrf import cli
sys.exit(cli.main(sys.argv[1:]))
