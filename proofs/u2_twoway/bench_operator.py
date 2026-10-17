"""CPU-only timing/memory diagnostics for one feedback field, not GPU speed."""
from pathlib import Path
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling.boundary_feedback import (
    _feedback_values, apply_feedback, build_feedback_weights,
    build_feedback_smoother, sm121_smooth, smdsm_smooth,
    _feedback_leaf,
)

assert jax.devices()[0].platform=="cpu"
weight=build_feedback_weights(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,
    parent_we=22,parent_sn=20,child_we=36,child_sn=30)
smoother=build_feedback_smoother(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,child_we=36,child_sn=30)
parent=jnp.zeros((44,20,22),dtype=jnp.float32)
child=jnp.asarray(np.random.default_rng(310).normal(size=(44,30,36)),dtype=jnp.float32)
result={"backend":"cpu","scope":"actual leaf JIT API versus prior eager leaf API; not GPU/coupled performance", "shape":list(child.shape),"cache":"NONE","arms":{}}
for option in (0,1,2):
    def run():
        return _feedback_leaf(parent,child,weight,smoother,option)
    start=time.perf_counter()
    jax.block_until_ready(run())
    compile_s=time.perf_counter()-start
    exe=_feedback_leaf.lower(parent,child,weight,smoother,option).compile()
    samples=[]
    for _ in range(15):
        start=time.perf_counter()
        out=run()
        jax.block_until_ready(out)
        samples.append(time.perf_counter()-start)
    memory=exe.memory_analysis()
    eager = _feedback_leaf.__wrapped__
    jax.block_until_ready(eager(parent,child,weight,smoother,option))
    eager_samples=[]
    for _ in range(15):
        start=time.perf_counter()
        value=eager(parent,child,weight,smoother,option)
        jax.block_until_ready(value)
        eager_samples.append(time.perf_counter()-start)
    result["arms"][str(option)]={"compile_s":compile_s,"median_s":float(np.median(samples)),
        "legacy_eager_median_s":float(np.median(eager_samples)),
        "eager_over_compiled_ratio_cpu_only":float(np.median(eager_samples)/np.median(samples)),
        "temporary_bytes":memory.temp_size_in_bytes,"arguments_bytes":memory.argument_size_in_bytes,
        "output_bytes":memory.output_size_in_bytes,"finite":bool(np.isfinite(np.asarray(out)).all())}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
