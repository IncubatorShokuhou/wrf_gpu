from pathlib import Path
import dataclasses
import jax.numpy as jnp

def real_atmosphere(case):
    # Use actual c33 atmospheric columns/vertical metrics. The tiny neutral
    # six-level construction produced nonfinite SW and is retained as failed
    # evidence; it cannot establish the real RUC producer contract.
    from gpuwrf.io.netcdf_lock import Dataset
    from test_v013_operational_smoke import _grid, _base_state
    with Dataset(case) as ds:
        nz=len(ds.dimensions['bottom_top'])
        grid=_grid(nz=nz,ny=2,nx=3)
        state=_base_state(grid)
        def field(name,ny=2,nx=3):
            return jnp.asarray(ds[name][0,...,20:20+ny,20:20+nx],jnp.float32)
        qv=field('QVAPOR');pressure=field('P')+field('PB');height=field('PH')+field('PHB')
        values=dict(u=field('U',2,4),v=field('V',3,3),theta=(field('T')+300.)*(1.+jnp.float32(461.6/287.)*qv),
            qv=qv,qc=field('QCLOUD'),qi=field('QICE'),qs=field('QSNOW'),qr=field('QRAIN'),qg=field('QGRAUP'),
            p=pressure,p_total=pressure,p_perturbation=field('P'),
            ph=height,ph_total=height,ph_perturbation=field('PH'),
            mu=field('MU')+field('MUB'),mu_total=field('MU')+field('MUB'),mu_perturbation=field('MU'),
            t_skin=field('TSK'),xland=field('XLAND'))
        state=state.replace(_cast=False,**values)
        metric={key.lower():jnp.asarray(ds[key][0],jnp.float32) for key in ('C1H','C2H','C3H','C4H','DNW','FNM','FNP')}
        metric['p_top']=jnp.asarray(ds['P_TOP'][0],jnp.float32)
        grid=dataclasses.replace(grid,metrics=dataclasses.replace(grid.metrics,**metric))
    return grid,state
