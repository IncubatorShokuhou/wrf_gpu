"""CPU-only compiled driver cost on original 44x42x42 Swiss boundary operands."""
import importlib.util,json,statistics,sys,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.assimilation.obs_driver import observation_driver,initial_observation_carry

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/v040/test_obs_full_coupled.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main(out):
    assert jax.devices()[0].platform=='cpu'
    inputs,meta=m.prepared_inputs();carry=initial_observation_carry(inputs,(44,42,42))
    names=('u','v','tb','qv','pbase','pp','z','uratx','vratx','tratx','kpbl','regime','pblh',
           'terrain','mut','muu','muv','c1h','c2h','msfux','msfuy','msfvx','msfvy','msftx','msfty')
    def fields(step):
        a=np.load(m.ROOT/f'obs_probe_{step:06d}.npz')
        kw={k:jnp.asarray(a[k]) for k in names}
        return dict(kw,ktau=jnp.int32(step),xtime=jnp.asarray(a['xtime']))
    f=jax.jit(lambda c,kw:observation_driver(c,inputs,dt_s=18.,dx=3000.,**kw))
    kw=fields(0);start=time.monotonic();compiled=f.lower(carry,kw).compile()
    compile_s=time.monotonic()-start
    refreshed,_=compiled(carry,kw);jax.block_until_ready(refreshed)
    times={}
    for label,c,values in (('refresh',carry,kw),('hold',refreshed,fields(1))):
        for _ in range(3):jax.block_until_ready(compiled(c,values))
        samples=[]
        for _ in range(25):
            start=time.monotonic();jax.block_until_ready(compiled(c,values));samples.append(time.monotonic()-start)
        times[label]=dict(median_s=statistics.median(samples),min_s=min(samples),samples_s=samples)
    memory=compiled.memory_analysis()
    result=dict(platform='cpu',shape=[44,42,42],reports=4,max_levels=3,ionf=2,compile_s=compile_s,
        timing=times,resident_carry_bytes=sum(x.nbytes for x in jax.tree.leaves(carry)),
        resident_input_bytes=sum(x.nbytes for x in jax.tree.leaves(inputs)),
        compiled_memory={k:getattr(memory,k) for k in ('argument_size_in_bytes','output_size_in_bytes',
                           'temp_size_in_bytes','alias_size_in_bytes')},
        optimized_f64_tokens=compiled.as_text().count('f64['),
        scope='component CPU timing only; no GPU performance/VRAM claim')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
