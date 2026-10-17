"""Fresh CPU init and CUDA-target lowering of the real Swiss CAM step (E156).

Optionally lower the original helpers on the same initialized operands first;
this is source regression evidence, never a replacement for the WRF oracle.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-ref", help="git ref supplying original CAM helpers")
    p.add_argument("--debug", action="store_true", help="retain original Python source-location aliases")
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import gpuwrf  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl, topologies
    from jax._src.pallas import pallas_call as pc
    import gpu_smoke_v034 as smoke

    original_call = pc.pallas_call
    native = [False]
    smoke._cpu_patches()

    def selected_call(*args, **kwargs):
        return original_call(*args, **{**kwargs, "interpret": not native[0]})

    pc.pallas_call = pl.pallas_call = selected_call
    backend = jax.default_backend
    jax.default_backend = lambda: "gpu" if native[0] else backend()
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.physics import ra_cam_lw_abs as absmod, ra_cam_lw as lwmod

    assert all(d.platform == "cpu" for d in jax.devices())
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = NestedPipelineConfig(smoke.CASE, a.out / "stream", a.out / "proof", hours=1, max_dom=1)
    _, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
    bundle = bundles["d01"]
    nml = replace(bundle.namelist, ra_lw_physics=3, ra_sw_physics=3)
    carry = smoke._rebuild_carry(carries["d01"], bundle.state, nml, ("radiation_diagnostics",))
    step = jnp.asarray(1, jnp.int32)
    clock = om.build_clock_base(nml)
    init_s = time.monotonic() - started
    topology = topologies.get_topology_desc(platform="cuda", target_config=a.target.read_text(), topology="1x1x1")
    sharding = jax.sharding.SingleDeviceSharding(topology.devices[0])

    def abstract(v):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            # Product arrays explicitly placed on the operational device are
            # committed; resident namelist/clock operands can be uncommitted.
            # Committing every abstract leaf adds non-product SDY constraints.
            return jax.ShapeDtypeStruct(v.shape, v.dtype, weak_type=bool(getattr(v, "weak_type", False)),
                                       sharding=sharding if getattr(v, "committed", False) else None)
        return v

    arguments = jax.tree.map(abstract, (carry, nml, step, clock))
    native[0] = True
    record = {"physical_backend": "cpu", "case": str(smoke.CASE), "init_s": init_s,
              "namelist": {"ra_lw_physics": 3, "ra_sw_physics": 3, "dt_s": nml.dt_s,
                           "radiation_cadence_steps": nml.radiation_cadence_steps}, "arms": {}}

    def lower(label):
        jax.clear_caches()
        t = time.monotonic()
        lowered = om._advance_chunk_fori.lower(*arguments, n_steps=1, cadence=int(nml.radiation_cadence_steps))
        raw = str(lowered.compiler_ir()).encode()
        (a.out / f"{label}.mlir.gz").write_bytes(gzip.compress(raw, mtime=0))
        if a.debug:
            debug = lowered.compiler_ir().operation.get_asm(enable_debug_info=True).encode()
            (a.out / f"{label}.debug.mlir.gz").write_bytes(gzip.compress(debug, mtime=0))
        record["arms"][label] = {"mlir_sha256": hashlib.sha256(raw).hexdigest(),
                                 "bytes": len(raw), "lower_s": time.monotonic() - t}
        (a.out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        print(label, record["arms"][label], flush=True)

    if a.base_ref:
        replacements = [(absmod, "src/gpuwrf/physics/ra_cam_lw_abs.py",
                         ("_interp32", "_h2o_overlap_tw", "_h2o_weights", "_trcabn")),
                        (lwmod, "src/gpuwrf/physics/ra_cam_lw.py", ("radclwmx_fluxes", "_cumsum_levels", "radems", "interp5", "trcems", "lookup_index"))]
        restored = []
        for mod, file, names in replacements:
            source = subprocess.check_output(["git", "-C", str(root), "show", f"{a.base_ref}:{file}"], text=True)
            parsed = ast.parse(source)
            for node in parsed.body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    restored.append((mod, node.name, getattr(mod, node.name)))
                    exec(compile(ast.Module(body=[node], type_ignores=[]), mod.__file__, "exec"), mod.__dict__)
        try:
            lower("base_fresh")
        finally:
            for mod, name, fn in restored:
                setattr(mod, name, fn)
    lower("candidate")


if __name__ == "__main__":
    main()
