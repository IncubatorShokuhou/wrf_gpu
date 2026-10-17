"""Changed caller arithmetic cost on CPU; no GPU/forecast speed claim."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('tree',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    sys.path.insert(0,str(a.tree/'src'))
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.grid import DycoreMetrics
    from gpuwrf.coupling.physics_couplers import (_rho_from_state,_temperature_from_theta,
        _dry_theta_view,_wrf_phy_prep_rho_from_state,_theta_m_tendency_from_dry)
    jax.config.update('jax_enable_x64',True)
    assert jax.devices()[0].platform=='cpu'
    report=dict(platform='cpu',scope='Changed arithmetic only, excluding the unchanged Tiedtke solver; no GPU/fullforecast speed claim',
                cache='NONE',rows=[],clock={})
    for ny,nx in ((42,42),(70,120),(117,267)):
        nz=44;shape=(nz,ny,nx);r=jnp.float32
        metrics=DycoreMetrics.flat(ny=ny,nx=nx,nz=nz,eta_levels=jnp.linspace(1.,0.,nz+1),
            top_pressure_pa=5000.,provenance='CPU changed-caller resource probe only')
        phi=jnp.broadcast_to(jnp.linspace(0.,12000.*9.81,nz+1,dtype=r)[:,None,None],(nz+1,ny,nx))
        inputs=(jnp.full(shape,300.,r),jnp.full(shape,.01,r),jnp.full(shape,80000.,r),phi,
                jnp.full((ny,nx),95000.,r),jnp.full(shape,.001,r),jnp.full(shape,-2.e-6,r))
        row=dict(shape=list(shape),arms={})
        for fixed in (False,True):
            def seam(theta,qv,p,ph,mu,rth,rqv):
                state=SimpleNamespace(theta=theta,qv=qv,p=p,ph=ph,ph_total=ph,mu=mu,mu_total=mu)
                T=_temperature_from_theta(_dry_theta_view(state) if fixed else theta,p)
                rho=_wrf_phy_prep_rho_from_state(state,metrics,output_dtype=theta.dtype) if fixed else _rho_from_state(state)
                rate=_theta_m_tendency_from_dry(rth,rqv,theta,qv,theta.dtype) if fixed else rth
                return T,rho,(theta+18.*rate).astype(theta.dtype)
            start=time.monotonic();compiled=jax.jit(seam).lower(*inputs).compile();compile_s=time.monotonic()-start
            jax.block_until_ready(compiled(*inputs));times=[]
            for _ in range(15):
                start=time.perf_counter();jax.block_until_ready(compiled(*inputs));times.append(time.perf_counter()-start)
            m=compiled.memory_analysis()
            row['arms']['fixed' if fixed else 'parent']=dict(compile_s=compile_s,warm_median_s=float(np.median(times)),
                arguments=m.argument_size_in_bytes,outputs=m.output_size_in_bytes,temporary=m.temp_size_in_bytes)
            del compiled;jax.clear_caches()
        report['rows'].append(row)
    for dynamic in (False,True):
        def clock(lead):
            return 1+jnp.rint(lead/18.).astype(jnp.int32) if dynamic else jnp.int32(60)
        compiled=jax.jit(clock).lower(jnp.float32(0)).compile();m=compiled.memory_analysis()
        report['clock']['fixed' if dynamic else 'parent']=dict(arguments=m.argument_size_in_bytes,
            outputs=m.output_size_in_bytes,temporary=m.temp_size_in_bytes,
            instructions=len(jax.make_jaxpr(clock)(jnp.float32(0)).jaxpr.eqns))
    report['peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
