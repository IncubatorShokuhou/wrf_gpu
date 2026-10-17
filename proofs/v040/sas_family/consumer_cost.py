"""Matched-input CPU allocation/timing of the changed consumer, not GPU cost."""
from collections import namedtuple
from pathlib import Path
import ast
import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling.scalesas_adapter import apply_scalesas_rates
from gpuwrf.coupling.native_sas_adapter import apply_native_sas_rates
from gpuwrf.coupling.physics_couplers import _add_a2c_u_increment, _add_a2c_v_increment

assert jax.devices()[0].platform=='cpu'
jax.config.update('jax_enable_x64',True)
class Fields(namedtuple('Fields','theta qv qc qi rainc_acc u v')):
    def replace(self,**kwargs):return self._replace(**kwargs)

shape=(44,70,120);surface=shape[1:]
full=lambda value:jnp.full(shape,value,jnp.float32)
state=Fields(full(300.),full(.01),full(0.),full(0.),jnp.zeros(surface,jnp.float32),
             jnp.full((44,70,121),3.,jnp.float32),jnp.full((44,71,120),4.,jnp.float32))
rates=(full(1.e-4),full(1.e-6),full(1.e-7),full(0.),full(1.e-8),full(0.),
       jnp.full(surface,.001,jnp.float32),full(.001),full(.002))
text=subprocess.check_output(['git','show','1a9da66b2:src/gpuwrf/coupling/native_sas_adapter.py'],text=True)
tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_native_sas_rates')
namespace=dict(jnp=jnp,REAL=jnp.float32,apply_scalesas_rates=apply_scalesas_rates,
               _add_a2c_u_increment=_add_a2c_u_increment,_add_a2c_v_increment=_add_a2c_v_increment)
exec(compile(ast.Module(body=[node],type_ignores=[]),'original_consumer','exec'),namespace)
report={'platform':'cpu','shape':list(shape),'dt':18.,'specified':True,'periodic_x':False,
        'producer_physics_changed':False,'GPU_launch_VRAM_cost':'unmeasured','arms':{}}
for name,fn in [('before',namespace['apply_native_sas_rates']),('after',apply_native_sas_rates)]:
    opts=dict(entry_state=state,periodic_x=False,periodic_y=False)
    if name=='after':opts['specified_or_nested']=True
    def consume(s,r):return fn(s,r,18.,**{**opts,'entry_state':s})
    start=time.monotonic();exe=jax.jit(consume).lower(state,rates).compile();compile_s=time.monotonic()-start
    jax.block_until_ready(exe(state,rates));samples=[]
    for _ in range(15):
        start=time.monotonic();jax.block_until_ready(exe(state,rates));samples.append(time.monotonic()-start)
    memory=exe.memory_analysis()
    report['arms'][name]=dict(compile_s=compile_s,warm_median_s=float(np.median(samples)),
          temp_bytes=memory.temp_size_in_bytes,argument_bytes=memory.argument_size_in_bytes,
          output_bytes=memory.output_size_in_bytes)
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
