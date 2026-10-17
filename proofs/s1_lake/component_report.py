"""CPU-only resource diagnostics for standalone lake initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import jax
from jax import numpy as jnp
import numpy as np

from gpuwrf.physics.lake_initialization import LakeInitInputs, lake_initial_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert jax.devices()[0].platform == "cpu"
    n = 128
    inp = LakeInitInputs(*(jnp.full((n,), value, jnp.int32 if idx < 2 else jnp.float32)
                          for idx, value in enumerate((17,8,100.,12.,260.,1.,0.,2.,10.))))
    start = time.perf_counter()
    fn = jax.jit(lake_initial_state).lower(inp).compile()
    compile_s = time.perf_counter()-start
    result = jax.block_until_ready(fn(inp))
    samples = []
    for _ in range(12):
        start = time.perf_counter()
        jax.block_until_ready(fn(inp))
        samples.append(time.perf_counter()-start)
    memory = fn.memory_analysis()
    leaves = [np.asarray(x) for x in jax.tree.leaves(result)]
    stats = {key: getattr(memory,key) for key in (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
        "alias_size_in_bytes", "generated_code_size_in_bytes")}
    source = Path(__file__).resolve().parents[2] / "src/gpuwrf/physics/lake_initialization.py"
    report = dict(backend="cpu", n_columns=n, compile_s=compile_s,
                  warm_median_s=statistics.median(samples), samples_s=samples,
                  compiled_memory=stats,
                  materialized_output_bytes=sum(x.nbytes for x in leaves),
                  payload_float32_values_per_column=sum(np.asarray(x).size for x in result.columns)//n,
                  nonfinite=sum(int((~np.isfinite(x)).sum()) for x in leaves),
                  payload_dtypes=sorted({str(np.asarray(x).dtype) for x in result.columns}),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  default_forecast_calls_added=0,
                  scope="Standalone init only; CPU compile/memory/wall diagnostics, no GPU/S1 claim")
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == "__main__":
    main()
