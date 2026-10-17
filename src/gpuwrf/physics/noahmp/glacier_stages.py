"""WRF glacier energy, phase and water in three bounded column kernels."""
from functools import partial
import jax.numpy as jnp
from gpuwrf.kernels.phys_noahmp_columns import column_call
from .glacier_energy import atmosphere,thermal,radiation,surface_flux,soil_temperature,TFRZ
from .glacier_phase import phase_change
from .glacier_water import water


def energy(inputs,*,opt_gla,opt_alb,opt_snf,opt_tbot,opt_stc):
    v={k:list(a) if isinstance(a,(list,tuple)) else a for k,a in inputs.items()}
    e,rho,direct,diffuse,sw=atmosphere(v)
    dz=[]
    for k,z in enumerate(v['zsnso']):
        prev=jnp.zeros_like(z) if k==0 else v['zsnso'][k-1]
        dz.append(jnp.where(k==v['isnow']+3,-z,prev-z))
    df,hc,fact=thermal(v,dz)
    radiation(v,direct,diffuse,opt_alb)
    surface_flux(v,dz,df,e,rho,opt_gla,opt_stc)
    soil_temperature(v,df,hc,opt_tbot,opt_stc)
    if opt_stc==2:v['tg']=jnp.where((v['snowh']>.05)&(v['tg']>TFRZ),TFRZ,v['tg'])
    v.update(__dz=dz,__fact=fact,__sw=sw,hcpct=hc)
    return v


def phase(inputs,*,opt_gla,**unused):
    v={k:list(a) if isinstance(a,(list,tuple)) else a for k,a in inputs.items()}
    v['__imelt']=phase_change(v,v['__dz'],v['__fact'],opt_gla)
    del v['__fact']
    return v


def hydrology(inputs,*,opt_gla,opt_snf,**unused):
    v={k:list(a) if isinstance(a,(list,tuple)) else a for k,a in inputs.items()}
    water(v,v['__dz'],v['__imelt'],opt_gla,opt_snf)
    empty=(v['snowh']<=1.e-6)|(v['sneqv']<=1.e-3)
    v['snowh']=jnp.where(empty,0.,v['snowh']);v['sneqv']=jnp.where(empty,0.,v['sneqv'])
    sw=v['__sw'];v['albedo']=jnp.where(sw!=0.,v['fsr']/jnp.where(sw!=0.,sw,1.),-999.9)
    return v


def staged_kernel(inputs,config,*,history=False):
    from gpuwrf.coupling.glacier_surface_hook import _OUTPUTS,_HISTORY
    options={key:getattr(config,key) for key in ('opt_gla','opt_alb','opt_snf','opt_tbot','opt_stc')}
    shape=(1,len(config.indices));carry=inputs
    for name,fn in [('energy',energy),('phase',phase),('water',hydrology)]:
        carry=column_call(partial(fn,**options),(carry,),grid_shape=shape,name='noahmp_glacier_'+name,block=32,warps=1)
    return {k:carry[k] for k in (*_OUTPUTS,*(_HISTORY if history else ()))}
