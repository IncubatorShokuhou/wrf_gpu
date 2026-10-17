"""Fresh-process byte gate for real CLI streams, terminal carry and 3 formats."""
import hashlib
import json
import os
from pathlib import Path
import sys

assert os.environ['JAX_PLATFORMS'] == 'cpu'
import gpuwrf
import jax
import numpy as np
from gpuwrf.contracts import state as state_contract
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.restart import read_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry
from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state
from gpuwrf.runtime.restart_store import RestartStore


def hashes(tree):
    leaves, definition = jax.tree.flatten(tree)
    records = []
    for leaf in leaves:
        value = np.asarray(leaf)
        if value.dtype.kind in 'fc':
            assert np.isfinite(value).all()
        records.append([value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()])
    return {'treedef': str(definition), 'leaves': records}

state_contract._gpu_device = lambda: jax.devices('cpu')[0]
root = Path(sys.argv[1])
control = json.loads((root / 'control_carry.json').read_text())
resume = json.loads((root / 'resume_carry.json').read_text())
assert control['own_steps'] == resume['own_steps'] == {'d01': 200}
assert control['hashes'] == resume['hashes']
cut = json.loads((root / 'cut_generation.json').read_text())
store = RestartStore(root / 'checkpoints', 8 * 1024**3, 2, 0)
snapshot, manifest = store.read(cut['generation'], expected_identity=cut['identity'])
assert hashes(snapshot['carries']) == cut['hashes']
formats = {
    'pickle': read_restart(root / 'formats/restart.pkl')[0],
    'netcdf': read_wrfrst_carry(root / 'formats/wrfrst.nc')[0],
    'checkpoint': read_checkpoint_with_runtime_state(root / 'formats/checkpoint.pkl')[4],
}
format_leaves = {}
for name, carry in formats.items():
    actual = hashes({'d01': carry})
    assert actual == cut['hashes'], name
    format_leaves[name] = len(actual['leaves'])
names = sorted(p.name for p in (root / 'control').glob('wrfout_*'))
assert len(names) == 2
assert names == sorted(p.name for p in (root / 'stream').glob('wrfout_*'))
variables = 0
for name in names:
    with Dataset(root / 'control' / name) as a, Dataset(root / 'stream' / name) as b:
        a.set_auto_mask(False)
        b.set_auto_mask(False)
        assert a.ncattrs() == b.ncattrs()
        for key in a.ncattrs():
            assert np.array_equal(a.getncattr(key), b.getncattr(key)), key
        assert list(a.variables) == list(b.variables)
        for key in a.variables:
            x, y = np.asarray(a.variables[key][...]), np.asarray(b.variables[key][...])
            assert (x.dtype, x.shape, x.tobytes()) == (y.dtype, y.shape, y.tobytes()), (name, key)
            variables += 1
report = {'status': 'PASS', 'platform': 'cpu', 'gpu_executed': False,
          'files': names, 'variables_byte_exact': variables,
          'terminal_carry_leaves_byte_exact': len(control['hashes']['leaves']),
          'standalone_format_leaves_byte_exact': format_leaves,
          'numerical_scope': 'restart/determinism regression; not WRF fidelity',
          'harness': 'State device guard redirected to CPU; product stepping, solver and CLI'}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
