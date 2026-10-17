"""Original complete ME and public Jensen wrapper; operational admission pending."""
import jax
import jax.numpy as jnp
from .math import _real
from .processes import processes
from .sedimentation import sedimentation
from .finish import finish,last_active_density
from . import _native
from .level_support import _get


def _me(fields,tables,xp):
    # Host readers stay NumPy-only; convert once at the trace boundary so
    # dynamic lookup indices never index a NumPy array during JAX tracing.
    tables={n:a if getattr(a,'_jensen_table_ref',False) else xp.asarray(a) for n,a in tables.items()}
    p=processes(fields,tables,xp=xp)
    held=last_active_density(p,xp=xp)
    p['sedi']=(xp.max(p['sedi'].astype('int32'),axis=-1)!=0) if _native.active.get() else xp.any(p['sedi'],axis=-1)
    # Both arms of the original main k loop reset these accumulators to zero.
    p['qrpre1d']=xp.zeros(p['qv'].shape[:-1],dtype=p['qv'].dtype)
    p['qipre1d']=xp.zeros(p['qv'].shape[:-1]+(4 if _native.active.get() else 3,),dtype=p['qv'].dtype)
    p=sedimentation(p,xp=xp)
    p=finish(p,tables,held,xp=xp)
    names=('theta','qv','qc','qr','nr','qi','ni','ai','ci','effc1d','effi1d','dbz1d',
           'icetype1d','vmi1d','di1d','rhopo1d','phii1d','qrpre1d','qipre1d',
           'sedimentation_steps','held_rhobar')
    return {n:p[n] for n in names}


def me_ishmael(fields,tables,*,xp=jnp):
    if xp is jnp:
        # Original gammln REAL*8 is the sole DOUBLE island. Every REAL
        # literal/operand in the generated processes retains its carry dtype.
        with jax.enable_x64(True):return _me(fields,tables,xp)
    return _me(fields,tables,xp)


def mp_jensen_ishmael(fields,tables,dt,it,*,rain=0.,snow=0.,xp=jnp):
    """Full public fields use vertical-last layout; effective radii are metres."""
    _,r=_real(fields['th'],xp)
    f=dict(theta=fields['th'],pres_e=fields['p'],dzmic=fields['dz'],dt=r(dt),it=xp.asarray(it,dtype='int32'))
    for n in ('qv','qc','qr','nr'):f[n]=fields[n]
    for n in ('qi','ni','ai','ci'):f[n]=xp.stack(tuple(fields[n+str(i)] for i in (1,2,3)),axis=-1)
    zero=xp.zeros_like(fields['qv']);zero_cat=xp.zeros_like(f['qi'])
    f.update(effc1d=zero,effi1d=zero,dbz1d=xp.full_like(zero,r(-35.)),qrpre1d=zero,qipre1d=zero_cat)
    for n in ('icetype1d','vmi1d','di1d','rhopo1d','phii1d'):f[n]=zero_cat
    out=me_ishmael(f,tables,xp=xp)
    return public_outputs(fields,out,rain,snow,xp=xp)


def public_outputs(fields,out,rain,snow,*,xp=jnp):
    _,r=_real(fields['th'],xp)
    result={n:fields[n] for n in ('p','dz')}
    result['th']=out['theta']
    for n in ('qv','qc','qr','nr'):result[n]=out[n]
    for i in range(3):
        for n in ('qi','ni','ai','ci'):result[n+str(i+1)]=_get(out[n],(i,)) if _native.active.get() else out[n][...,i]
        for target,source in (('vm','vmi1d'),('di','di1d'),('rho','rhopo1d'),('phi','phii1d'),('type','icetype1d')):
            result[target+str(i+1)]=_get(out[source],(i,)) if _native.active.get() else out[source][...,i]
    result['effc']=xp.maximum(xp.minimum(out['effc1d'],r(50.e-6)),r(2.51e-6))
    result['effi']=xp.maximum(xp.minimum(out['effi1d'],r(999.e-6)),r(5.e-6))
    result['dbz']=xp.maximum(out['dbz1d'],r(-35.))
    a,b,c=(_get(out['qipre1d'],(i,)) if _native.active.get() else out['qipre1d'][...,i] for i in range(3));water=out['qrpre1d']
    # Original left association also applies when existing accumulators are
    # nonzero; adding a pre-summed step total would round differently.
    precip=dict(snow=r(snow)+a+b+c,snowstep=a+b+c,
                rain=r(rain)+water+a+b+c,rainstep=water+a+b+c)
    return result,precip


def step(fields,dt,it,*,rain=0.,snow=0.):
    """Operational kernel entry: the selected host tables become trace constants."""
    from .tables import get_tables
    return mp_jensen_ishmael(fields,get_tables(fields['th'].dtype),dt,it,rain=rain,snow=snow)
