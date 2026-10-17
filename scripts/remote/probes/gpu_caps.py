#!/usr/bin/env python3
"""wz1080 GPU capability probe (one process per call; run under wz-gpu-lock).

python gpu_caps.py --section env|bench|tridiag|pallas|cmdbuf [--out result.json]

cmdbuf: set XLA_FLAGS="--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE" (what gpuwrf injects by default,
C25) in the environment BEFORE this process starts to probe CUDA-graph conditional nodes on this GPU.
Every section records its error text instead of raising, so one call yields one matrix row.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback


def _err(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) < 1500 else text[:1500] + " ..."


def _timeit(fn, *args, reps: int = 7):
    out = fn(*args)
    out.block_until_ready() if hasattr(out, "block_until_ready") else None
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], out


def section_env(jax, jnp) -> dict:
    d = jax.devices()[0]
    res = {
        "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__,
        "backend": jax.default_backend(),
        "platform": d.platform,
        "device_kind": getattr(d, "device_kind", None),
        "compute_capability": getattr(d, "compute_capability", None),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    try:
        res["memory_stats"] = {k: v for k, v in (d.memory_stats() or {}).items() if "limit" in k or "in_use" in k}
    except Exception as exc:  # noqa: BLE001
        res["memory_stats"] = _err(exc)
    try:
        from jax._src.lib import cuda_path  # resolved once at import (E107)
        res["cuda_path"] = cuda_path
        ptxas = os.path.join(cuda_path or "", "bin", "ptxas")
        if os.path.exists(ptxas):
            res["ptxas"] = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception as exc:  # noqa: BLE001
        res["cuda_path"] = _err(exc)
    if shutil.which("nvidia-smi"):
        res["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    x = jnp.arange(1024.0)
    res["jit_ok"] = float(jax.jit(lambda v: (v * v).sum())(x)) == float((x * x).sum())
    return res


def section_bench(jax, jnp) -> dict:
    res: dict = {}
    n = 64 * 1024 * 1024
    for dt in ("float32", "float64"):
        try:
            x = jnp.ones((n,), dtype=dt)
            t, _ = _timeit(jax.jit(lambda v: v * 2.0 + 1.0), x)
            res[f"stream_{dt}_GBps"] = round(2 * x.nbytes / t / 1e9, 1)
        except Exception as exc:  # noqa: BLE001
            res[f"stream_{dt}_GBps"] = _err(exc)
    for dt, m in (("float32", 4096), ("float64", 2048)):
        try:
            a = jax.random.normal(jax.random.PRNGKey(0), (m, m), dtype=dt)
            t, _ = _timeit(jax.jit(lambda p: p @ p), a, reps=5)
            res[f"gemm_{dt}_TFLOPs"] = round(2 * m ** 3 / t / 1e12, 3)
        except Exception as exc:  # noqa: BLE001
            res[f"gemm_{dt}_TFLOPs"] = _err(exc)

    def chain(v):  # 64 dependent FMAs per element: compute-bound, stencil-like arithmetic intensity
        y = v
        for i in range(64):
            y = y * 0.999 + 0.001 * i
        return y

    for dt in ("float32", "float64"):
        try:
            x = jnp.ones((16 * 1024 * 1024,), dtype=dt)
            t, _ = _timeit(jax.jit(chain), x)
            res[f"fma_chain_{dt}_GFLOPs"] = round(2 * 64 * x.size / t / 1e9, 1)
        except Exception as exc:  # noqa: BLE001
            res[f"fma_chain_{dt}_GFLOPs"] = _err(exc)
    f32, f64 = res.get("fma_chain_float32_GFLOPs"), res.get("fma_chain_float64_GFLOPs")
    if isinstance(f32, float) and isinstance(f64, float) and f64:
        res["fp32_over_fp64_fma"] = round(f32 / f64, 1)
    return res


def section_tridiag(jax, jnp) -> dict:
    res: dict = {}
    nz, ncol = 44, 31239  # PROD d02 columns x levels, like MYNN's batched gtsv (A56)
    for dt in ("float32", "float64"):
        try:
            k = jax.random.PRNGKey(1)
            dl = jnp.zeros((ncol, nz), dt).at[:, 1:].set(-0.3)
            du = jnp.zeros((ncol, nz), dt).at[:, :-1].set(-0.3)
            d = jnp.ones((ncol, nz), dt) * 2.0
            b = jax.random.normal(k, (ncol, nz, 1), dtype=dt)
            f = jax.jit(jax.lax.linalg.tridiagonal_solve)
            t, x = _timeit(f, dl, d, du, b)
            ax = d[..., None] * x
            ax = ax.at[:, 1:].add(dl[:, 1:, None] * x[:, :-1]).at[:, :-1].add(du[:, :-1, None] * x[:, 1:])
            res[f"gtsv_{dt}"] = {"ms": round(t * 1e3, 3), "max_resid": float(jnp.abs(ax - b).max())}
        except Exception as exc:  # noqa: BLE001
            res[f"gtsv_{dt}"] = _err(exc)
    return res


def section_pallas(jax, jnp) -> dict:
    from jax.experimental import pallas as pl

    res: dict = {"JAX_PALLAS_USE_MOSAIC_GPU": os.environ.get("JAX_PALLAS_USE_MOSAIC_GPU")}

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0 + 1.0

    x = jnp.arange(1024, dtype=jnp.float32)
    for interp in (False, True):
        key = "triton_compiled" if not interp else "interpret_on_gpu"
        try:
            f = jax.jit(lambda v: pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
                                                 interpret=interp)(v))
            t0 = time.perf_counter()
            y = f(x).block_until_ready()
            res[key] = {"ok": bool(jnp.allclose(y, x * 2 + 1)), "first_call_s": round(time.perf_counter() - t0, 3)}
        except Exception as exc:  # noqa: BLE001
            res[key] = {"ok": False, "error": _err(exc)}
    return res


def section_cmdbuf(jax, jnp) -> dict:
    res: dict = {"xla_flags": os.environ.get("XLA_FLAGS", "")}

    def body(c):
        i, v = c
        return i + 1, jnp.where(v.sum() > 0, v * 0.5 + 1.0, v)

    try:
        f = jax.jit(lambda v: jax.lax.while_loop(lambda c: c[0] < 50, body, (0, v))[1])
        t, y = _timeit(f, jnp.ones((4096,), jnp.float32))
        res["while_cond_ok"] = bool(jnp.isfinite(y).all())
        res["while_cond_ms"] = round(t * 1e3, 3)
    except Exception as exc:  # noqa: BLE001
        res["while_cond_ok"] = False
        res["error"] = _err(exc)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=["env", "bench", "tridiag", "pallas", "cmdbuf"])
    ap.add_argument("--out")
    ap.add_argument("--allow-cpu", action="store_true", help="local code-path check only (no capability claim)")
    args = ap.parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    import jax
    import jax.numpy as jnp

    row = {"section": args.section, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        if not args.allow_cpu:
            assert jax.devices()[0].platform == "gpu", f"not a GPU process: {jax.devices()}"
        row["platform"] = jax.devices()[0].platform
        row["result"] = globals()[f"section_{args.section}"](jax, jnp)
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001
        row["ok"] = False
        row["error"] = _err(exc)
        row["traceback"] = traceback.format_exc()[-3000:]
    text = json.dumps(row, indent=1, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
