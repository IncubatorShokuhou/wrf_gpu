"""Changed-artifact CPU cost/memory census; never a GPU performance claim."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import statistics
import time
from types import SimpleNamespace

import jax
import numpy as np
from gpuwrf.io.stochastic_config import bind_stochastic
from gpuwrf.physics.stochastic_driver import initial_stochastic,advance_stochastic


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert jax.devices()[0].platform=="cpu"
    jax.config.update("jax_enable_x64",True)
    rows=[]
    for vertical in (0,1):
        controls=dict(skebs=1,sppt=1,spp=1,rand_perturb=1,nens=7,
            skebs_vertstruc=vertical,sppt_vertstruc=vertical,rand_pert_vertstruc=vertical,
            vertstruc_spp_conv=vertical,vertstruc_spp_pbl=vertical,vertstruc_spp_lsm=vertical)
        controls.update({"lengthscale_"+s:15000. for s in
            ("sppt","rand_pert","spp_conv","spp_pbl","spp_lsm")})
        grid=SimpleNamespace(nx=42,ny=42,nz=44,dx=3000.,dy=3000.)
        config=bind_stochastic({"stoch":controls},"d01")
        start=time.perf_counter()
        carry=initial_stochastic(config,grid,18,datetime(2023,1,15))
        jax.block_until_ready(carry)
        init=time.perf_counter()-start
        step=jax.jit(lambda c:advance_stochastic(c,config,dx=grid.dx,dy=grid.dy))
        program=str(jax.make_jaxpr(step)(carry))
        assert "f64[" not in program and "c128[" not in program
        start=time.perf_counter()
        compiled=step.lower(carry).compile()
        compile_s=time.perf_counter()-start
        carry=compiled(carry)
        jax.block_until_ready(carry)
        times=[]
        for _ in range(11):
            start=time.perf_counter()
            carry=compiled(carry)
            jax.block_until_ready(carry)
            times.append(time.perf_counter()-start)
        memory=compiled.memory_analysis()
        rows.append(dict(vertical=vertical,initialization_s=init,compile_s=compile_s,
            warm_step_median_s=statistics.median(times),warm_step_min_s=min(times),
            resident_bytes=sum(np.asarray(a).nbytes for a in jax.tree.leaves(carry)),
            compiled_memory={k:getattr(memory,k) for k in
                ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")},
            raw_float64_tokens=0))
        jax.clear_caches()
    result=dict(backend="cpu",GPU_execution=False,shape="Swiss-sized42x42x44; full43x43FFT; dt18",
        work="all7produced patterns,11warmsteps aftercompile+onewarm; no weather physics",
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,records=rows,
        limits="CPU changed-artifact accounting only; GPU launches/timing/peak/E41 pending")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
