"""Subprocess body for tests/test_w44_cycling_wrf_witness.py (not collected by pytest).

Evaluates mynn_coldstart_init_columns(..., initialize_qke, return_moments=True) on the
frozen original-WRF witness inputs exactly as proofs/w44_warmstart/compare_mynn_init.py
does (per-dx jit, REAL32 scope, mask = max(2*tke) < 2e-4). XLA_FLAGS must be set by the
caller BEFORE this process imports JAX. Writes candidate arrays only; the caller asserts.
usage: python mynn_init_runner.py <witness.npz> <out.npz>
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

import gpuwrf
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, mynn_coldstart_init_columns


def evaluate(scalars, arrays):
    candidate = np.empty(arrays.shape[:2] + (3,), np.float32)
    mask_all = np.empty(arrays.shape[0], bool)
    for dx in sorted(set(scalars[:, -1])):
        indices = np.flatnonzero(scalars[:, -1] == dx)
        a, s = arrays[indices], scalars[indices]
        value = lambda slot: jnp.asarray(a[:, :, slot])
        z = jnp.zeros_like(value(0))
        state = MynnPBLColumnState(value(0), value(1), value(2), value(3), value(4), value(5), value(6), value(7),
                                   value(8), z, z, z, qc=value(9), qi=value(10), qs=value(11), qsq=z, qc_bl=z,
                                   qi_bl=z, cldfra_bl=z, exner=value(13))
        mask = np.max(a[:, :, 5] * np.float32(2), axis=-1) < np.float32(.0002)
        with jax.enable_x64(False):
            f = jax.jit(lambda state, ust, land, mask: mynn_coldstart_init_columns(
                state, ust, float(dx), land, initialize_qke=mask, return_moments=True))
            result = f(state, jnp.asarray(s[:, 0]), jnp.asarray(s[:, 4]), jnp.asarray(mask))
            jax.block_until_ready(result)
        qke, _pblh, el, qsq = map(np.asarray, result)
        candidate[indices, :, 0] = qke
        candidate[indices, :, 1] = qsq
        candidate[indices, :, 2] = el
        mask_all[indices] = mask
    return candidate, mask_all


def main():
    witness, out = sys.argv[1:3]
    data = np.load(witness)
    tags = sorted({key.split("__")[1] for key in data.files if key.startswith("mynn__")})
    result = {}
    for tag in tags:
        candidate, mask = evaluate(data[f"mynn__{tag}__scalars"], data[f"mynn__{tag}__arrays"])
        result[f"{tag}__candidate"] = candidate
        result[f"{tag}__mask"] = mask
    meta = {"backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "gpuwrf_file": os.path.realpath(gpuwrf.__file__), "tags": tags,
            "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset>")}
    np.savez(out, meta=np.array(json.dumps(meta)), **result)


if __name__ == "__main__":
    main()
