"""CPU component compile/memory/timing receipt; no GPU speed claim."""
import argparse
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.io.windfarm import WindFarm
from gpuwrf.physics.wind_fitch import dragforce


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert jax.devices()[0].platform=='cpu'
    shape=(20,40,80);base=jnp.full(shape,8.,jnp.float32)
    dz=jnp.full(shape,20.,jnp.float32)
    z=jnp.broadcast_to(jnp.arange(21,dtype=jnp.float32)[:,None,None]*20, (21,40,80))
    curve=((2.,.9,0.),(4.,.85,60.),(8.,.8,500.),(12.,.7,1600.),(20.,.4,2000.),(25.,.1,2000.))
    report={'backend':'cpu','scope':'component only; no GPU/runtime readiness or roofline claim','shape':shape,'arms':[]}
    for nt in (1,10,100):
        farm=WindFarm(tuple((t%80,(t//80)%40,(80.,80.,.12,2000.,curve)) for t in range(nt)))
        fn=jax.jit(lambda u,v,z,dz,q,du,dv:dragforce(u,v,z,dz,q,du,dv,dx=3000.,dt=18.,farm=farm))
        vals=(base,base*.25,z,dz,base*.1,base*0,base*0)
        start=time.monotonic();lowered=fn.lower(*vals);exe=lowered.compile();compile_s=time.monotonic()-start
        memory=exe.memory_analysis()
        exe(*vals)[0].block_until_ready();times=[]
        for _ in range(10):
            start=time.monotonic();jax.block_until_ready(exe(*vals));times.append(time.monotonic()-start)
        row=dict(nt=nt,compile_s=compile_s,median_s=float(np.median(times)),
                 argument_bytes=memory.argument_size_in_bytes,output_bytes=memory.output_size_in_bytes,
                 temp_bytes=memory.temp_size_in_bytes,hlo_chars=len(lowered.as_text()),
                 analysis='compact (nt,max_curve_rows) constants; no nt*domain*levels temporary; sequential turbines preserve collision order')
        report['arms'].append(row);print(row,flush=True)
        jax.clear_caches()
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
