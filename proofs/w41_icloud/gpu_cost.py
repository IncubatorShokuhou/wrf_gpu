"""GPU per-call cost of wrf_cal_cldfra3 at the WN3 0227 domain shapes (runs inside the arm's lock session).

usage: python gpu_cost.py CASE OUT.json
The product calls cal_cldfra3 once per radiation call and domain, in the REAL context (x64 off, f32
operands). Operands: the 640 real fixture columns tiled over each domain's (ny, nx); timing is
value-insensitive (no data-dependent loops). Records median device-synchronized wall of 20 calls after
one warm call, compile time and the compiled memory analysis.
"""
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.physics import cloud_fraction as cf

CASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
# W41_COST_DRY_RUN=1: CPU logic check only (the JSON records the platform; never a GPU claim).
assert jax.devices()[0].platform == "gpu" or os.environ.get("W41_COST_DRY_RUN") == "1", jax.devices()
fixture = np.load(Path(__file__).resolve().parent / "fixtures/icloud3_columns_v1.npz")
out = {"platform": jax.devices()[0].platform, "device": str(jax.devices()[0]), "domains": {}}
for d in ("d01", "d02", "d03"):
    with Dataset(CASE / "input" / f"wrfinput_{d}") as ds:
        ny, nx, nz = (len(ds.dimensions[k]) for k in ("south_north", "west_east", "bottom_top"))
        dx, dy = float(ds.DX), float(ds.DY)
    assert fixture["qv"].shape[-1] == nz, (fixture["qv"].shape, nz)
    idx = np.arange(ny * nx) % fixture["qv"].shape[0]
    with jax.enable_x64(False):
        cols = [jnp.asarray(fixture[k][idx].reshape(ny, nx, nz), jnp.float32)
                for k in ("qv", "qc", "qi", "qs", "dz", "p", "t")]
        xland = jnp.asarray(fixture["xland"][idx].reshape(ny, nx), jnp.float32)
        gridkm = cf.wrf_gridkm(dx, dy)
        fn = jax.jit(lambda *a: cf.wrf_cal_cldfra3(*a, gridkm))
        t0 = time.perf_counter()
        compiled = fn.lower(*cols, xland).compile()
        compile_s = time.perf_counter() - t0
        jax.block_until_ready(compiled(*cols, xland))
        walls = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(compiled(*cols, xland))
            walls.append(time.perf_counter() - t0)
    mem = compiled.memory_analysis()
    out["domains"][d] = {
        "shape": [ny, nx, nz], "gridkm": float(gridkm), "compile_s": round(compile_s, 2),
        "call_ms_median": round(1e3 * float(np.median(walls)), 4), "call_ms_min": round(1e3 * min(walls), 4),
        "temp_bytes": int(mem.temp_size_in_bytes), "argument_bytes": int(mem.argument_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
    }
    print(d, out["domains"][d], flush=True)
OUT.write_text(json.dumps(out, indent=1) + "\n")
