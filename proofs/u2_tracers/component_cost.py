"""CPU-only source/sink cost on Swiss geometry; no GPU/S1/fidelity claim."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

root,out=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root/"src"))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.contracts.state import State
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.tracers.config import TracerSpecies,UserTracerConfig,Emission
from gpuwrf.tracers.runtime import sources_and_sinks
from gpuwrf.validation.tier2 import make_ideal_grid

assert jax.devices()[0].platform=="cpu"
grid=make_ideal_grid(44,42,42,dx_m=3000.,dy_m=3000.)
state=State.zeros(grid)
state=state.replace(mu_total=jnp.full_like(state.mu_total,90000),
    ph_total=jnp.broadcast_to(jnp.arange(45)[:,None,None]*981.,state.ph_total.shape),
    rain_acc=jnp.full_like(state.rain_acc,.25),rainc_acc=jnp.zeros_like(state.rainc_acc))
base=OperationalNamelist.from_grid(grid,dt_s=18.)
rows=[]
for count,source_count in ((1,0),(1,2),(8,2)):
    species=tuple(TracerSpecies(f"TRACER{i}",.0001,.005,.1) for i in range(count))
    emissions=tuple(e for i in range(count) for e in
        (Emission(i,0.,100.,20,21,20,21,1.,0.,True),
         Emission(i,0.,100.,10,30,10,30,0.,1e-6,False))) if source_count else ()
    nl=replace(base,tracer_config=UserTracerConfig(species,emissions,"a"*64))
    s=state.replace(tracers=jnp.full((count,44,42,42),1e-5,jnp.float32),
                    tracer_budget=jnp.zeros((count,4),jnp.float32))
    def f(value):
        before=value.replace(rain_acc=jnp.zeros_like(value.rain_acc),rainc_acc=jnp.zeros_like(value.rainc_acc))
        result=sources_and_sinks(value,before,nl,jnp.float32(0))
        return result.tracers,result.tracer_budget
    start=time.perf_counter()
    program=jax.jit(f).lower(s).compile()
    compile_s=time.perf_counter()-start
    for _ in range(3):
        jax.block_until_ready(program(s))
    samples=[]
    for _ in range(15):
        start=time.perf_counter();result=program(s);jax.block_until_ready(result)
        samples.append(time.perf_counter()-start)
    assert all(np.isfinite(np.asarray(x)).all() for x in result)
    stats=program.memory_analysis()
    rows.append(dict(species=count,emissions_per_species=source_count,compile_s=compile_s,
        median_s=float(np.median(samples)),min_s=min(samples),max_s=max(samples),
        compiled_memory_bytes={key:getattr(stats,key) for key in
            ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")},
        persistent_tracer_bytes=int(s.tracers.nbytes+s.tracer_budget.nbytes)))
    out.write_text(json.dumps(dict(backend="cpu",scope="source/sink component only; synthetic positive operands on Swiss geometry, not WRF fidelity or GPU performance",cache="disabled",rows=rows),indent=2)+"\n")
    print(rows[-1],flush=True)
