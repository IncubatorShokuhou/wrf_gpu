"""Reuse the real CLI byte gate and record the actual surface wind activity."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1])
surface = int(sys.argv[2])
sys.argv = [sys.argv[0], str(root)]
runpy.run_path(str(Path(__file__).parents[1] / 't_restartfix/verify_cli.py'))

import numpy as np
from gpuwrf.runtime.restart_store import RestartStore

cut = json.loads((root / 'cut_generation.json').read_text())
assert cut['identity']['temf_config']['d01']['sf_sfclay_physics'] == surface
store = RestartStore(root / 'checkpoints', 8 * 1024**3, 2, 0)
snapshot, _ = store.read(cut['generation'], expected_identity=cut['identity'])
carry = snapshot['carries']['d01']
shape = carry.state.theta.shape[-2:]
winds = {}
for name in ('u10e', 'v10e'):
    value = getattr(carry.history_diagnostics, name, None)
    if surface in (2, 4):
        assert value is not None, name
        value = np.asarray(value)
        assert value.shape == shape and value.dtype == np.float32, name
        assert np.isfinite(value).all(), name
        winds[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                           nonzero=int(np.count_nonzero(value)),
                           minimum=float(value.min()), maximum=float(value.max()),
                           sha256=hashlib.sha256(value.tobytes()).hexdigest())
    else:
        assert value is None, name
        winds[name] = None
report_path = root / 'verification.json'
report = json.loads(report_path.read_text())
report.update(surface_scheme=surface, step_100_winds=winds,
              control_steps=200, sigkill_steps=100, resumed_steps=200)
report_path.write_text(json.dumps(report, indent=2) + '\n')
inventory = {}
for directory in ('control', 'stream', 'checkpoints', 'formats'):
    for path in sorted((root / directory).rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            inventory[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,
                                                        sha256=digest)
(root / 'artifact_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
print(json.dumps({'surface_scheme': surface, 'step_100_winds': winds}))
