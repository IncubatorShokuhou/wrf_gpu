"""Changed accumulator artifact cost on CPU, not a GPU/forecast speed claim."""
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import NamedTuple

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/"src"))
import jax
import jax.numpy as jnp
from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import scan_adapters as module

jax.config.update("jax_enable_x64",True)
assert jax.devices()[0].platform == "cpu"
old_source=subprocess.check_output(["git","show","0013f53df:src/gpuwrf/coupling/scan_adapters.py"],cwd=root,text=True)
node=next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef) and n.name=="_apply_mp_replacements")
namespace=dict(module.__dict__)
exec(compile(ast.Module(body=[node],type_ignores=[]),"<original-accumulator>","exec"),namespace)
old=namespace["_apply_mp_replacements"]


class Surface(NamedTuple):
    rain_acc: object
    snow_acc: object
    graupel_acc: object
    ice_acc: object
    hail_acc: object
    re_cloud: object = None
    re_ice: object = None
    re_snow: object = None

    def replace(self,**kwargs):
        return self._replace(**kwargs)


result=dict(platform="cpu",gpu_executed=False,scope=__doc__,cache="NONE GPUWRF_JAX_CACHE=0",shapes={})
prior_source=subprocess.check_output(["git","show","fe481a6ed:src/gpuwrf/coupling/scan_adapters.py"],cwd=root,text=True)
prior_node=next(n for n in ast.parse(prior_source).body if isinstance(n,ast.FunctionDef) and n.name=="_apply_mp_replacements")
prior_namespace=dict(module.__dict__)
exec(compile(ast.Module(body=[prior_node],type_ignores=[]),"<prior-split-accumulator>","exec"),prior_namespace)
prior=prior_namespace["_apply_mp_replacements"]
for ny,nx in ((42,42),(70,120),(117,267)):
    state=Surface(*(jnp.ones((ny,nx),jnp.float32) for _ in range(5)))
    amounts=tuple(jnp.full((ny*nx,),x,jnp.float32) for x in (8.,1.,2.,0.,.5))
    row={}
    for label,fn in (("old",old),("new",module._apply_mp_replacements)):
        def run(s,*incs):
            return fn(s,PhysicsTendency(accumulator_increments=dict(zip(Surface._fields[:5],incs))),ny=ny,nx=nx,nz=1)
        t0=time.perf_counter()
        compiled=jax.jit(run).lower(state,*amounts).compile()
        compile_s=time.perf_counter()-t0
        jax.block_until_ready(compiled(state,*amounts))
        values=[]
        for _ in range(40):
            t0=time.perf_counter()
            actual=compiled(state,*amounts)
            jax.block_until_ready(actual)
            values.append(time.perf_counter()-t0)
        memory=compiled.memory_analysis()
        hlo_text=compiled.as_text()
        row[label]=dict(compile_s=compile_s,warm_median_us=statistics.median(values)*1.e6,
                         argument_bytes=memory.argument_size_in_bytes,output_bytes=memory.output_size_in_bytes,
                         temporary_bytes=memory.temp_size_in_bytes,alias_bytes=memory.alias_size_in_bytes,
                         optimized_f64_array_tokens=hlo_text.count("f64["),
                         hlo_sha256=hashlib.sha256(hlo_text.encode()).hexdigest())
        # Disjoint-channel reconstruction is a separate physical accounting
        # check, not equality to the intentionally incorrect baseline.
        if label == "new":
            assert float(actual.rain_acc[0,0]) == 5.5
    result["shapes"][f"{ny}x{nx}"]=row
    # The one-hour CPU diagnostic emitted x64-disabled cast warnings. Prove
    # the final carry-kind cleanup has the identical component program in
    # that REAL context; this is not a whole-GPU-program identity claim.
    if ny == 42:
        hlo=[]
        with jax.enable_x64(False):
            for fn in (prior,module._apply_mp_replacements):
                def run(s,*incs):
                    return fn(s,PhysicsTendency(accumulator_increments=dict(zip(Surface._fields[:5],incs))),ny=ny,nx=nx,nz=1)
                lowered=jax.jit(run).lower(state,*amounts)
                hlo.append(lowered.compiler_ir("hlo").as_hlo_text())
        result["real_kind_component_identity"] = dict(equal=hlo[0]==hlo[1],
            prior_sha256=hashlib.sha256(hlo[0].encode()).hexdigest(),
            final_sha256=hashlib.sha256(hlo[1].encode()).hexdigest(),
            scope="accumulator component under CPU REAL context; prior fe481a6ed vs final carry-kind cleanup")
        assert hlo[0] == hlo[1]
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
