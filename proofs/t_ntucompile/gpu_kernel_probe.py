"""Kernel-only GPU probe: XLA compile + execute of ONE production column kernel.

Separates the XLA-integrated Triton compile of the NTU/Jensen pallas_call from
the whole-program compile. Fields are physically plausible constant columns
(Swiss-like 44 levels, 1764 columns); this is a compile/runtime probe, not a
fidelity check. Must run under scripts/with_gpu_lock.sh.
usage: gpu_kernel_probe.py ntu|jensen OUT_JSON
"""
import json
import os
import sys
import time

scheme, out = sys.argv[1], sys.argv[2]
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
t0 = time.monotonic()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
assert jax.devices()[0].platform == 'gpu', jax.devices()
ncol, nz = 1764, 44
z = np.linspace(0, 1, nz, dtype=np.float32)
col = lambda v: np.broadcast_to(np.asarray(v, np.float32), (ncol, nz)).copy()
base = dict(th=col(285 + 40 * z), p=col(95000 * np.exp(-2.3 * z) + 2000), dz=col(100 + 900 * z))
if scheme == 'ntu':
    from gpuwrf.kernels.phys_ntu import full_column
    from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES
    from gpuwrf.physics.ntu.column import PROGNOSTICS
    fields = dict(base, pii=col((base['p'] / 1e5) ** (2 / 7)))
    fields.update({n: col(0.) for n in PROGNOSTICS})
    fields['qv'] = col(8e-3 * (1 - z))
    fields['w'] = np.zeros((ncol, nz + 1), np.float32)
    fields.update({n: np.zeros((ncol,), np.float32) for n in ('rainnc', 'snownc', 'grapnc', 'hailnc')})
    fn = jax.jit(lambda f, d: full_column(f, d, output_names=OUTPUT_NAMES))
    args = (fields, np.float32(18))
else:
    from gpuwrf.kernels.phys_jensen import full_column
    from gpuwrf.physics.jensen_ishmael.registry import output_mapping
    from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_mapping
    mapping = output_mapping()
    names = [n for n in mapping if n not in ('effc', 'effi')]
    fields = dict(base, **{n: col(0.) for n in names})
    fields['qv'] = col(8e-3 * (1 - z))
    mapping.update(ice_diagnostic_mapping(55))
    outs = (*mapping, 'th')
    fn = jax.jit(lambda f, d, i: full_column(f, d, i, output_names=outs))
    args = (fields, np.float32(18), np.int32(1))
t = time.monotonic()
lowered = fn.lower(*args)
lower_s = time.monotonic() - t
t = time.monotonic()
compiled = lowered.compile()
compile_s = time.monotonic() - t
times = []
for _ in range(3):
    t = time.monotonic()
    res = compiled(*args)
    jax.block_until_ready(res)
    times.append(time.monotonic() - t)
leaves = jax.tree_util.tree_leaves(res)
finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves)
mem = compiled.memory_analysis()
r = dict(scheme=scheme, platform=jax.devices()[0].platform, device=str(jax.devices()[0]), ncol=ncol, nz=nz,
         lower_s=lower_s, compile_s=compile_s, exec_s=times, finite=finite, outputs=len(leaves),
         temp_bytes=getattr(mem, 'temp_size_in_bytes', None), wall_s=time.monotonic() - t0,
         scope='kernel-only XLA compile/execute on GPU, synthetic plausible columns; not fidelity')
open(out, 'w').write(json.dumps(r, indent=2) + '\n')
print(json.dumps(r))
