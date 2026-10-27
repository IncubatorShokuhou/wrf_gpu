"""Deviceless Triton lowering of the actual FP32 ray kernel (CPU, no CUDA)."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax._src.pallas.triton import lowering
from gpuwrf.backends import pallas as pl
from gpuwrf.kernels.topographic_shadow import topographic_shadow_native


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    assert jax.devices()[0].platform == "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    original = pl.cuda_native
    results = []
    try:
        # Only choose the CUDA lowering branch of this local facade. No GPU
        # backend is initialized or executable compiled/run.
        pl.cuda_native = lambda: True
        for nested in (False, True):
            shape = jax.ShapeDtypeStruct((93, 111), jnp.float32)
            fn = lambda a, b, c, d: topographic_shadow_native(a, b, c, d,
                dx_m=1000., dy_m=1000., shadow_length_m=25000., nested=nested)
            jp = jax.make_jaxpr(fn)(shape, shape, shape, shape).jaxpr
            params = next(e.params for e in jp.eqns if e.primitive.name == "pallas_call")
            result = lowering.lower_jaxpr_to_triton_module(params["jaxpr"], params["grid_mapping"], "cuda", 120)
            ir = str(result.module)
            (args.out / ("nested.ttir" if nested else "root.ttir")).write_text(ir)
            results.append({"nested": nested, "lowering": "PASS", "IR_characters": len(ir),
                            "forbidden_f64_present": "f64" in ir})
            assert "f64" not in ir
    finally:
        pl.cuda_native = original
    (args.out / "receipt.json").write_text(json.dumps({"backend": "cpu", "GPU_executed": False,
        "target": "cuda/sm120", "block_size": 128, "num_warps": 4, "results": results}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
