"""Measure the compiled Fitch hook, including mass-wind and height preparation."""
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace
import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.io.windfarm import WindFarm
from gpuwrf.coupling.windfarm_hook import windfarm_sources


class Fields(NamedTuple):
    u: object
    v: object
    ph: object
    qke: object

    def replace(self,**updates):return self._replace(**updates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert jax.devices()[0].platform=='cpu'
    nz,ny,nx,nt=20,40,80,100
    curve=((2.,.9,0.),(4.,.85,60.),(8.,.8,500.),(12.,.7,1600.),(20.,.4,2000.),(25.,.1,2000.))
    farm=WindFarm(tuple((t%nx,t//nx,(81.,80.,.12,2000.,curve)) for t in range(nt)))
    n=SimpleNamespace(dt_s=18.,grid=SimpleNamespace(projection=SimpleNamespace(dx_m=3000.)),windfarm=farm)
    def call(u,v,ph,q,du,dv):
        state=Fields(u,v,ph,q)
        out,a,b,power=windfarm_sources(state,state,du,dv,n)
        return out.qke,a,b,power
    u=jnp.full((nz,ny,nx+1),8.,jnp.float32);v=jnp.full((nz,ny+1,nx),2.,jnp.float32)
    ph=jnp.broadcast_to((jnp.arange(nz+1,dtype=jnp.float32)*20*9.81)[:,None,None],(nz+1,ny,nx))
    q=jnp.full((nz,ny,nx),.5,jnp.float32);zero=jnp.zeros_like(q)
    values=(u,v,ph,q,zero,zero)
    start=time.monotonic();lowered=jax.jit(call).lower(*values);exe=lowered.compile();compile_s=time.monotonic()-start
    out=exe(*values);jax.block_until_ready(out);assert all(np.isfinite(np.asarray(a)).all() for a in out)
    times=[]
    for _ in range(10):
        start=time.monotonic();jax.block_until_ready(exe(*values));times.append(time.monotonic()-start)
    memory=exe.memory_analysis()
    report=dict(backend='cpu',scope='Fitch hook including actual preparation; not full model or GPU peak memory',
                shape=[nz,ny,nx],nt=nt,compile_s=compile_s,median_s=float(np.median(times)),
                argument_bytes=memory.argument_size_in_bytes,output_bytes=memory.output_size_in_bytes,
                temp_bytes=memory.temp_size_in_bytes,optimized_hlo_chars=len(exe.as_text()),
                f64_type_mentions=exe.as_text().count('f64['),new_power_carry_bytes=ny*nx*4)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
