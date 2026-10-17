"""NTU advance_small: small_dt runs only when a level merges its result (t-ntucompile).

Every small_dt output is merged as where(m9 & alive, out, old); with no selected
level nothing it returns is read, so skipping it is exact (WRF's IF around the
CALL). On the heaviest real c88 columns 97-99 % of the small steps have no
selected level (NP12), which is why the column kernel was slow.
"""
import re
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from gpuwrf.physics.ntu import support

ROOT = Path(__file__).resolve().parents[2]


def test_generated_small_dt_call_is_masked_by_its_merge_mask():
    text = (ROOT / 'src/gpuwrf/physics/ntu/phases_generated.py').read_text()
    lines = text.splitlines()
    calls = [i for i, l in enumerate(lines) if 'call_out = ' in l and re.search(r'\bsmall_dt\b', l)]
    assert len(calls) == 1
    call = lines[calls[0]]
    assert call.strip().startswith('call_out = masked_call(xp.logical_and(m9, alive), small_dt, ')
    merges = []
    for line in lines[calls[0] + 1:]:
        if 'call_out' not in line:
            break
        merges.append(re.search(r"xp\.where\(xp\.logical_and\((\w+), alive\), xp\.asarray\(call_out\[", line).group(1))
    assert len(merges) == 29 and set(merges) == {'m9'}
    rest = '\n'.join(lines[calls[0] + 1 + len(merges):])
    assert 'call_out' not in rest.split('\ndef ')[0]


def _routine(d, *, xp=np, c=None):
    calls.append(1)
    return {'dts': d['dts'] * np.float32(2) + np.float32(1), 'n': d['n'] + np.int32(1)}


calls = []


def test_masked_call_selects_routine_only_when_a_lane_is_selected():
    d = {'dts': np.arange(4, dtype=np.float32), 'n': np.int32(3)}
    calls.clear()
    out = support.masked_call(np.zeros(4, bool), _routine, d, xp=np)
    assert calls and out['dts'].tolist() == [1., 3., 5., 7.]   # literal reference always calls

    f = jax.jit(lambda sel, d: support.masked_call(sel, _routine, d, xp=jnp))
    with jax.enable_x64(True):
        some = f(jnp.array([False, True, False, False]), d)
        none = f(jnp.zeros(4, bool), d)
    assert np.asarray(some['dts']).tolist() == [1., 3., 5., 7.] and int(some['n']) == 4
    assert np.asarray(none['dts']).tolist() == [0., 0., 0., 0.] and int(none['n']) == 0   # skipped: never merged
    assert some['dts'].dtype == none['dts'].dtype == jnp.float32
