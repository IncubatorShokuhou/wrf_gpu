"""Complete WRF Noah-MP glacier timestep for OPT_GLA=1/2.

The column body uses fixed register lists; GPU execution is one Pallas call.
No host/device transfer and no grid-wide vertical or Newton iteration loops.
Pristine reference: WRF 4.7.1 phys/module_sf_noahmp_glacier.F (REAL throughout).
"""
from __future__ import annotations
import jax.numpy as jnp
from .glacier_energy import atmosphere, thermal, radiation, surface_flux, soil_temperature, TFRZ
from .glacier_phase import phase_change
from .glacier_water import water


def glacier_column(inputs, *, opt_gla=1, opt_alb=2, opt_snf=1, opt_tbot=2, opt_stc=1, xp=jnp):
    """List-valued scalar/SIMD column kernel; also accepts NumPy for literal debugging."""
    if opt_gla not in (1,2):
        raise ValueError('OPT_GLA must be 1 or 2')
    v={k:list(a) if isinstance(a,(list,tuple)) else a for k,a in inputs.items()}
    eair,rho,direct,diffuse,sw=atmosphere(v,xp)
    dz=[]
    for k,z in enumerate(v['zsnso']):
        prev=xp.zeros_like(z) if k==0 else v['zsnso'][k-1]
        dz.append(xp.where(k==v['isnow']+3,-z,prev-z))
    df,hc,fact=thermal(v,dz,xp)
    radiation(v,direct,diffuse,opt_alb,xp)
    surface_flux(v,dz,df,eair,rho,opt_gla,opt_stc,xp)
    soil_temperature(v,df,hc,opt_tbot,opt_stc,xp)
    if opt_stc==2:
        v['tg']=xp.where((v['snowh']>.05)&(v['tg']>TFRZ),TFRZ,v['tg'])
    imelt=phase_change(v,dz,fact,opt_gla,xp)
    v['hcpct']=hc
    water(v,dz,imelt,opt_gla,opt_snf,xp)
    empty=(v['snowh']<=1.e-6)|(v['sneqv']<=1.e-3)
    v['snowh']=xp.where(empty,0.,v['snowh']);v['sneqv']=xp.where(empty,0.,v['sneqv'])
    v['albedo']=xp.where(sw!=0.,v['fsr']/xp.where(sw!=0.,sw,1.),-999.9)
    return v
