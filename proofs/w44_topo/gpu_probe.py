"""Single-process native GPU oracle + bounded warm operator timing.

No GPU child processes, no profiling launcher, no messages or data writes
outside the lane's artifact/cache directories. --dry-run imports no JAX.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fixture = Path(manifest["fixture"])
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(fixture) as data:
        for row in manifest["rows"]:
            for name in ("terrain", "old_height", "coszen", "hrang", "mask", "height"):
                value = data[row["key"] + "_" + name]
                assert np.isfinite(value).all(), (row["key"], name)
    if args.dry_run:
        assert "jax" not in sys.modules, "dry-run must stay JAX-dark"
        print(json.dumps({"dry_run": "PASS", "cases": len(manifest["rows"]), "jax_imported": False}))
        return
    smoke = os.environ.get("TOPO_LAUNCH_SMOKE") == "1"
    if smoke:
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1"
        assert os.environ["JAX_PLATFORMS"] == "cpu"
    else:
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1", "official GPU lock required"
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.topographic_shading import toposhad_solar_geometry_from_seconds, toposhad_solar_azimuth
    from gpuwrf.kernels.topographic_shadow import topographic_shadow_native, topographic_solar_geometry_native
    from gpuwrf.backends import pallas as pl
    assert jax.devices()[0].platform == ("cpu" if smoke else "gpu"), jax.devices()
    assert smoke or pl.cuda_native(), "native CUDA Pallas branch required"
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"platform": jax.devices()[0].platform, "device": str(jax.devices()[0]),
               "fixture_sha256": manifest["fixture_sha256"], "rows": [],
               "geometry": "GPU WRF REAL toposhad clock, no EOT", "benchmark_scope": "isolated kernel+geometry, host dispatch+sync included",
               "cache_dir": os.environ.get("GPUWRF_JAX_CACHE_DIR")}
    functions = {}
    with np.load(fixture) as data:
        for row in manifest["rows"]:
            key = row["key"]
            host = lambda name: data[key + "_" + name]
            get = lambda name: jnp.asarray(host(name), jnp.float32)
            group = (row["domain"], row["nested"])
            if group not in functions:
                def one(terrain, old, lat, lon, sina, cosa, declin, midpoint_seconds,
                        dx=row["dx_m"], dy=row["dy_m"], nested=row["nested"]):
                    if smoke:
                        csza, hrang = topographic_solar_geometry_native(lat, lon, declin,
                            midpoint_seconds, jnp.float32(0), jnp.float32(0), interpret=True, caller_units=True)
                    else:
                        csza, hrang = toposhad_solar_geometry_from_seconds(lat, lon, declin,
                            midpoint_seconds, jnp.float32(0))
                    azi = toposhad_solar_azimuth(lat, csza, declin, hrang, sina, cosa)
                    return topographic_shadow_native(terrain, old, csza, azi,
                        dx_m=dx, dy_m=dy, shadow_length_m=25000., nested=nested, interpret=smoke)
                functions[group] = jax.jit(one)
            inputs = tuple(get(name) for name in ("terrain", "old_height", "latitude", "longitude", "sina", "cosa", "declin")) + (jnp.float32((row["minute"] + 15.) * 60.),)
            began = time.perf_counter()
            output = jax.block_until_ready(functions[group](*inputs))
            cold_wall = time.perf_counter() - began
            mask, height = (np.asarray(x) for x in output)
            assert np.isfinite(height).all() and np.isfinite(mask).all()
            try:
                np.testing.assert_array_equal(mask, host("mask"))
                np.testing.assert_allclose(height, host("height"), atol=.01, rtol=2.e-5)
            except AssertionError:
                # The v9 failure log kept only five cells. Retain the complete
                # failed output and exact host operands for CPU localization.
                np.savez_compressed(args.out / (key + "_failure.npz"),
                    **{name: host(name) for name in ("terrain", "old_height", "latitude", "longitude",
                        "sina", "cosa", "declin", "coszen", "hrang", "mask", "height")},
                    actual_mask=mask, actual_height=height, minute=np.float32(row["minute"]))
                raise
            # CPU smoke checks the oracle only; timing repetitions are GPU-only.
            warmups, repeats = (0, 1) if smoke else (3, 20)
            for _ in range(warmups):
                jax.block_until_ready(functions[group](*inputs))
            samples = []
            for _ in range(repeats):
                began = time.perf_counter()
                jax.block_until_ready(functions[group](*inputs))
                samples.append(time.perf_counter() - began)
            result = dict(row, mask_misses=0, max_height_error_m=float(np.max(np.abs(height - host("height")))),
                          first_call_wall_s=cold_wall, warm_median_s=float(np.median(samples)),
                          warm_min_s=float(np.min(samples)), warm_max_s=float(np.max(samples)))
            receipt["rows"].append(result)
            (args.out / "bench.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    receipt["memory_stats"] = jax.devices()[0].memory_stats()
    receipt["status"] = "CPU_LAUNCH_SMOKE_PASS" if smoke else "PASS"
    receipt["scientific_GPU_gate"] = "PENDING" if smoke else "operator-only"
    (args.out / "bench.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
