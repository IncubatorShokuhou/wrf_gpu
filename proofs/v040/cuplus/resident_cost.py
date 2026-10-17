"""CPU artifact cost for the actual resident CuP caller, not GPU timing."""
from pathlib import Path
import argparse,hashlib,json,os,statistics,sys,time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.kernels.phys_cuplus_driver import column_step,initial_column
    fixture=args.source/'proofs/v040/cuplus/full_callers'
    manifest=json.loads((fixture/'manifest.json').read_text())
    file=fixture/'full_callers.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==manifest['fixture_sha256']
    row=next(r for r in manifest['rows'] if r['group']=='cup' and r['step']==1 and r['case']==1)
    with np.load(file,allow_pickle=False) as bank:
        raw={key:np.array(bank[row['key']+'_'+key]) for key in row['profiles']+row['scalars']}
    names=('T','QV','P','DZ','RHO','W0AVG','U','V','TH','Z','ZW','PI')
    arrays=tuple(jnp.asarray(raw[key],jnp.float32) for key in names)
    nz=len(raw['T']);w=jnp.zeros(nz+1,jnp.float32);w0=jnp.zeros(nz,jnp.float32)
    held=initial_column(10,nz)
    ctl=jnp.asarray([54.,9000.,1500.,6.,150.,1e-4,1.,0.],jnp.float32)
    def call(a,w,w0,h,c,step):
        return column_step(a,w,w0,h,c,step,scheme=10,stepcu=5,cudt=0.,
            num_bins=5,th_bin_size=.1,r_bin_size=1e-4,min_deep_freq=.1,
            min_shallow_freq=.001,interpret=True)
    started=time.monotonic()
    lower=jax.jit(call).lower(arrays,w,w0,held,ctl,jnp.int32(1))
    stable=lower.as_text();compiled=lower.compile();optimized=compiled.as_text()
    compile_s=time.monotonic()-started
    w0new,fresh=jax.block_until_ready(compiled(arrays,w,w0,held,ctl,jnp.int32(1)))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(fresh))
    times={}
    for label,h in [('fresh',held),('held',dict(fresh,NCA=jnp.float32(108.)))]:
        samples=[]
        for _ in range(5):
            tick=time.monotonic()
            jax.block_until_ready(compiled(arrays,w,w0new,h,ctl,jnp.int32(1)))
            samples.append((time.monotonic()-tick)*1000.)
        times[label]=dict(median_ms=statistics.median(samples),samples_ms=samples)
    memory=compiled.memory_analysis()
    record=dict(scope=__doc__,platform=jax.default_backend(),gpu_executed=False,
        source=str(args.source),fixture_sha256=manifest['fixture_sha256'],compile_s=compile_s,
        stable_f64_tokens=stable.count('f64'),optimized_f64_tokens=optimized.count('f64'),
        timing=times,memory={name:getattr(memory,name) for name in
            ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')},
        source_sha256={str(file.relative_to(args.source)):hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted((args.source/'src/gpuwrf/kernels').glob('phys_*cup*.py'))},
        execution='one per-column Pallas program; resident NCA skips fresh solve inside program; CPU interpretation only',
        caveats='No GPU launch count, STACK, VRAM or GPU speed claim. Compile includes interpreted CPU control flow.')
    assert record['platform']=='cpu'
    assert record['optimized_f64_tokens']==0,'CuP original arithmetic is REAL only'
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
