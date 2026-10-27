"""Family selection shared by cold preparation, root replay and real-column smoke."""
import hashlib
import json
from pathlib import Path
import subprocess


def defaults(family):
    if family == 'diffusion':
        return dict(c_s=.25, diff_6th_factor=.12, dampcoef=.2, zdamp=5000.,
                    w_alpha=.3, w_beta=1., w_crit_cfl=1.)
    from gpuwrf.runtime_parameters import MYNN
    return {name: spec[0] for name, spec in MYNN.items()}


def bind(namelist, family, values=None):
    from gpuwrf.runtime_parameters import bind_diffusion_parameters, bind_mynn_parameters
    fn = bind_diffusion_parameters if family == 'diffusion' else bind_mynn_parameters
    return fn(namelist, defaults(family) if values is None else values)


def cpu_operands(source, family, variant):
    import jax
    import jax.numpy as jnp
    import numpy as np
    values = None if variant == 'off' else {k: jnp.float32(v) for k,v in defaults(family).items()}
    if family == 'mynn':
        from gpuwrf.physics import mynn_pbl as P
        from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
        fixture = source/'tests/v025/b_phys/fixtures/mynn_predict_saw_wrf.json'
        row = json.loads(fixture.read_text())['rows'][0]
        def array(value):
            if value is None:
                return None
            a = jnp.asarray(value, jnp.float32)
            return a[:4] if a.ndim and a.shape[0] == 12 else a
        state = P.MynnPBLColumnState(**{k:array(v) for k,v in row['state'].items()})
        surface = SurfaceFluxes(**{k:array(v) for k,v in row['flux'].items()})
        def program(inputs, controls):
            kwargs = {} if controls is None else {'sqfac':controls['sqfac'],
                'mynn_coefficients':{k:v for k,v in controls.items() if k != 'sqfac'}}
            return P.step_mynn_pbl_column_with_pblh(inputs[0], row['dt'], surface=inputs[1],
                edmf=True, dx=row['dx'], **kwargs)
        operands = ((state, surface), values)
        work = dict(fixture=str(fixture), columns=4, levels=44, dt=row['dt'], dx=row['dx'])
    else:
        from netCDF4 import Dataset
        from gpuwrf.kernels.dyn_diff_fp32 import sixth_order_fp32
        from gpuwrf.dynamics.explicit_diffusion import horizontal_deformation_2d, smag2d_horizontal_km
        fixture = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfinput_d01')
        with Dataset(fixture) as ds:
            q = np.asarray(ds['T'][0,:8,:16,:16],np.float32)
            mu = np.asarray(ds['MU'][0,:16,:16]+ds['MUB'][0,:16,:16],np.float32)
            c1,c2 = (np.asarray(ds[n][0,:8],np.float32) for n in ('C1H','C2H'))
            mx,my = (np.asarray(ds[n][0,:16,:16],np.float32) for n in ('MAPFAC_MX','MAPFAC_MY'))
            u,v = np.asarray(ds['U'][0,:8,:16,:17],np.float32),np.asarray(ds['V'][0,:8,:17,:16],np.float32)
            dx,dy = float(ds.DX),float(ds.DY)
        inputs = jax.tree.map(lambda x:jnp.asarray(x,jnp.float32),(q,mu,c1,c2,mx,my,u,v))
        def program(inputs, controls):
            selected = defaults('diffusion') if controls is None else controls
            q,mu,c1,c2,mx,my,u,v = inputs
            sixth = sixth_order_fp32(q,mu,c1,c2,mx,my,name='m',dt=18.,
                                      factor=selected['diff_6th_factor'],interpret=True)
            deformation = horizontal_deformation_2d(u,v,dx_m=dx,dy_m=dy)
            smag = smag2d_horizontal_km(*deformation,dx_m=dx,dy_m=dy,c_s=selected['c_s'],msftx=mx,msfty=my)
            return sixth,smag
        operands = (inputs,values)
        work = dict(fixture=str(fixture), shape=[8,16,16], dt=18., dx=dx, dy=dy,
                    scope='Real Swiss sixth-order and Smagorinsky operators; not full root or all damping consumers')
    pin = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    key = hashlib.sha256(('launch-smoke-'+pin+family+variant).encode()).hexdigest()
    return jax.jit(program),operands,key,work
