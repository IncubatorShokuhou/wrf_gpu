"""Observe existing init intermediates to localize eager/JIT association error.

Observation may change fusion; report that explicitly, never use this as a
production fidelity gate. The WS11 original-WRF witness remains fixed.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_pbl as P

NAMES = 'sqv init_dqw dtv gm gh sm sh qkw el qke pdk pdq b2l qsq b1l pblh'.split()


def main():
    oracle, production, output = map(Path, sys.argv[1:4])
    node = ast.parse(inspect.getsource(P.mynn_coldstart_init_columns)).body[0]
    assert isinstance(node.body[-2], ast.If) and node.body[-2].test.id == 'return_moments'
    node.body[-2].body[0].value = ast.Dict(keys=[ast.Constant(n) for n in NAMES],
                                         values=[ast.Name(id=n, ctx=ast.Load()) for n in NAMES])
    ast.fix_missing_locations(node)
    namespace = dict(P.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), P.__file__, 'exec'), namespace)
    observed = namespace['mynn_coldstart_init_columns']
    lines = (oracle / 'input.txt').read_text().splitlines()
    nz, nb, _, _ = map(float, lines[0].split()); nz, nb = int(nz), int(nb)
    scalars, arrays, offset = [], [], 1
    for _ in range(nb):
        scalars.append(list(map(float, lines[offset].split()))); offset += 1
        arrays.append([list(map(float, r.split())) for r in lines[offset:offset+nz]]); offset += nz
    scalars, arrays = np.asarray(scalars, np.float32), np.asarray(arrays, np.float32)
    captures = {}
    for dx in sorted(set(scalars[:, -1])):
        indices = np.flatnonzero(scalars[:, -1] == dx); a, s = arrays[indices], scalars[indices]
        value = lambda slot: jnp.asarray(a[:, :, slot])
        z = jnp.zeros_like(value(0))
        state = P.MynnPBLColumnState(value(0), value(1), value(2), value(3), value(4), value(5), value(6), value(7), value(8), z, z, z,
                qc=value(9), qi=value(10), qs=value(11), qsq=z, qc_bl=z, qi_bl=z, cldfra_bl=z, exner=value(13))
        mask = np.max(a[:, :, 5]*np.float32(2), axis=-1) < np.float32(.0002)
        f = lambda state, ust, land, mask: observed(state, ust, float(dx), land, initialize_qke=mask, return_moments=True)
        args = state, jnp.asarray(s[:, 0]), jnp.asarray(s[:, 4]), jnp.asarray(mask)
        with jax.enable_x64(False):
            compiled = jax.jit(f)(*args); jax.block_until_ready(compiled)
            with jax.disable_jit():
                eager = f(*args); jax.block_until_ready(eager)
        for mode, values in [('jit', compiled), ('eager', eager)]:
            for name, value in values.items():
                value = np.asarray(value)
                key = mode + '_' + name
                if key not in captures:
                    captures[key] = np.empty((nb, *value.shape[1:]), dtype=value.dtype)
                captures[key][indices] = value
    old = np.load(production)['values']; truth = np.load(oracle / 'expected.npz')['values']
    records = []
    for name in NAMES:
        a, b = captures['jit_'+name], captures['eager_'+name]
        error = np.abs(a.astype(float)-b.astype(float))
        pos = np.unravel_index(np.argmax(error), error.shape)
        records.append(dict(name=name, max_abs=float(error[pos]), pos=list(map(int,pos)),
                            jit=float(a[pos]), eager=float(b[pos]), differing=int(np.count_nonzero(a!=b))))
    qsq = captures['jit_qsq']; bound = 2e-12+2e-3*np.abs(truth[:, :, 1])
    metric = np.abs(qsq-truth[:, :, 1])/bound
    old_metric = np.abs(old[:, :, 1]-truth[:, :, 1])/bound
    worst = []
    for flat in np.argsort(old_metric.ravel())[-10:][::-1]:
        col, lev = np.unravel_index(flat, old_metric.shape)
        row = dict(col=int(col), lev=int(lev), original_qsq=float(truth[col,lev,1]),
                   production_qsq=float(old[col,lev,1]), old_bound_ratio=float(old_metric[col,lev]),
                   observed_bound_ratio=float(metric[col,lev]))
        for name in ('sqv','init_dqw','dtv','sh','qkw','el','qke','pdq','qsq'):
            row[name] = {mode:float(captures[mode+'_'+name][col,lev]) for mode in ('eager','jit')}
        worst.append(row)
    np.savez(output.with_suffix('.npz'), **captures)
    report = dict(records=records, worst=worst, observation_changes_qsq=int(np.count_nonzero(qsq!=old[:,:,1])),
                  observed_qsq_bound_ratio=float(metric.max()), production_qsq_bound_ratio=float(old_metric.max()),
                  helper_source_sha256=hashlib.sha256(inspect.getsource(P.mynn_coldstart_init_columns).encode()).hexdigest(),
                  scope='CPU intermediate observation only; fusion perturbed, not production fidelity')
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
