"""F2N04 diagnostic: where does the port sixth-order (wrf_sixth_order_scalar_tendf) differ from pristine
sixth_order_diffusion for WDM6 qs/qg?  Prints the worst cells, their 7-point stencils, and re-runs both
with subnormal inputs flushed to zero (E117 DAZ hypothesis).  CPU, light."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parents[1] / "tests/v025/b_diff"), str(HERE)]
assert os.environ.get("JAX_PLATFORMS") == "cpu"
import gpuwrf  # noqa: F401,E402
import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

import pristine_diffusion as PDF  # noqa: E402
from gpuwrf.dynamics import explicit_diffusion as ED  # noqa: E402

netCDF4.set_chunk_cache(4 * 2**20, 521, 0.75)
z = netCDF4.Dataset(sys.argv[1])
var = sys.argv[2]
g = lambda v: np.asarray(z[v][0], np.float32)  # noqa: E731
mut = g("MU") + g("MUB")
maps = {k: g(v) for k, v in (("msftx", "MAPFAC_MX"), ("msfty", "MAPFAC_MY"), ("msfux", "MAPFAC_UX"),
                              ("msfuy", "MAPFAC_UY"), ("msfvx", "MAPFAC_VX"), ("msfvy", "MAPFAC_VY"))}
dt = float(z.DT) / 3.0
lib = PDF.build(Path("<USER_HOME>/wrf_gpu2_lanes/o5-f2neg/build/sixth"))
tiny = np.finfo(np.float32).tiny


def both(q):
    port = np.asarray(ED.wrf_sixth_order_scalar_tendf(jnp.asarray(q), jnp.asarray(mut), c1=jnp.asarray(g("C1H")),
                      c2=jnp.asarray(g("C2H")), msftx=jnp.asarray(maps["msftx"]), msfty=jnp.asarray(maps["msfty"]),
                      dt=dt, diff_6th_factor=0.12, monotonic=True, specified_or_nested=True), np.float64)
    wrf = PDF.run(lib, q, mut, g("C1H"), g("C2H"), maps, name="m", dt=dt, factor=0.12, monotonic=True).astype(np.float64)
    return port, wrf


q = g(var)
port, wrf = both(q)
d = np.abs(port - wrf)
print(var, "max|d|", d.max(), "max|wrf|", np.abs(wrf).max(), "rel", d.max() / np.abs(wrf).max(),
      "cells d>1e-6*max", int((d > 1e-6 * np.abs(wrf).max()).sum()))
for flat in np.argsort(d.reshape(-1))[::-1][:4]:
    k, j, i = np.unravel_index(flat, d.shape)
    print(f" kji=({k},{j},{i}) port={port[k, j, i]:.6e} wrf={wrf[k, j, i]:.6e}")
    print("   q i-3..i+3:", " ".join(f"{x:.3e}" for x in q[k, j, max(i - 3, 0):i + 4]))
    print("   q j-3..j+3:", " ".join(f"{x:.3e}" for x in q[k, max(j - 3, 0):j + 4, i]))
qf = np.where(np.abs(q) < tiny, np.float32(0), q)
p2, w2 = both(qf)
print("subnormal inputs flushed:", int(((q != 0) & (np.abs(q) < tiny)).sum()),
      "-> max|d|", np.abs(p2 - w2).max(), "rel", np.abs(p2 - w2).max() / np.abs(w2).max())
neg = q < 0
print("negative inputs:", int(neg.sum()), "min", float(q.min()))
qp = np.maximum(q, 0)
p3, w3 = both(qp)
print("negatives clipped (diagnostic only):", "max|d|", np.abs(p3 - w3).max(), "rel", np.abs(p3 - w3).max() / np.abs(w3).max())
