"""Literal CPU reference for MP_NTU/NTU_MICRO, original level orientation.

This vector JAX/NumPy reference is not the GPU hot path. Native operation
uses a per-column Pallas entry so all adaptive loops stay in one kernel.
"""
import numpy as np
from . import phases_generated as phases
from .tables import selected_tables
from .support import any_true

PROGNOSTICS='qv qc qr qi qs qg qh nc nr ni ns ng nh qdcn qtcn qccn qrcn qnin fi fs vi vs vg ai as ag ah i3m'.split()
SURFACE='rainnc rainncv snownc snowncv grapnc grapncv hailnc hailncv sr'.split()
PRIVATE_PRECIP='clodncv icencv vincv vsncv vgncv fincv fsncv aincv asncv agncv ahncv i3mncv'.split()


def while_loop(condition,body,state,*,xp):
    if xp is np or getattr(xp,'literal',False):
        while np.any(condition(state)):state=body(state)
        return state
    import jax
    return jax.lax.while_loop(condition,body,state)


def flux(q,vt,rho,dz,dt,dtmn=None,*,xp=np):
    """Original PTFLUX/FLFLUX (F:3203–3263), top-down mass levels.

    Tendencies all read the old field, including the surface precipitation.
    The number/scalar variant deliberately omits the precipitation arithmetic.
    """
    r=np.dtype(q.dtype).type
    from . import _native
    native=_native.active.get()
    nstep=xp.maximum(np.int32(1),xp.max(xp.asarray(vt*dt/dz+r(1),dtype=np.int32),axis=-1))
    amount=xp.zeros(q.shape[:-1],dtype=q.dtype)
    denom=xp.asarray(nstep,dtype=q.dtype)
    def body(state):
        i,q,amount=state
        active=i<nstep
        if native:
            k=xp.arange(q.shape[-1],dtype=np.int32)
            other=(_native.previous(vt)*_native.previous(rho)*_native.previous(q)-vt*rho*q)/(dz*rho)
            dqdt=xp.where(k==0,(-vt*q)/dz,other)
        else:
            top=(-vt[...,:1]*q[...,:1])/dz[...,:1]
            other=(vt[...,:-1]*rho[...,:-1]*q[...,:-1]-vt[...,1:]*rho[...,1:]*q[...,1:])/(dz[...,1:]*rho[...,1:])
            dqdt=xp.concatenate((top,other),axis=-1)
        if dtmn is not None:
            if native:
                last=lambda x:_native.point(x,_native.levels.get()-1)
                next_amount=amount+last(vt)*last(rho)*last(q)*dtmn*r(60)/denom
            else:
                next_amount=amount+vt[...,-1]*rho[...,-1]*q[...,-1]*dtmn*r(60)/denom
            amount=xp.where(active,next_amount,amount)
        updated=q+dqdt*dt/denom[...,None]
        q=xp.where(active[...,None],updated,q)
        return i+np.int32(1),q,amount
    state=while_loop(lambda s:any_true(s[0]<nstep,xp=xp),body,(np.int32(0),q,amount),xp=xp)
    return state[1],state[2]


def step(fields,dt,*,ccnty=2,xp=np,observer=None,output_names=None):
    """Original MP caller: dry theta, full p/Exner, dz, staggered W.

    Arrays have (..., mass-level) and W has one extra interface. The public
    orientation is bottom-up; only the column's private work is top-down.
    """
    if xp is np or getattr(xp,'literal',False):
        return _step(fields,dt,ccnty=ccnty,xp=xp,observer=observer,output_names=output_names)
    import jax
    with jax.enable_x64(True):
        return _step(fields,dt,ccnty=ccnty,xp=xp,observer=observer,output_names=output_names)


def _step(fields,dt,*,ccnty=2,xp=np,observer=None,output_names=None):
    th=xp.asarray(fields['th']);r=np.dtype(th.dtype).type
    c=selected_tables(th.dtype,ccnty)
    dt=xp.asarray(dt,dtype=th.dtype);dtmn=dt/r(60)
    pii=xp.asarray(fields['pii'],dtype=th.dtype)
    w=xp.asarray(fields['w'],dtype=th.dtype)
    v={'tk3d':(th*pii)[...,::-1], 'p3d':xp.asarray(fields['p'],dtype=th.dtype)[...,::-1],
       'dz3d':xp.asarray(fields['dz'],dtype=th.dtype)[...,::-1],
       'w3d':(r(.5)*(w[...,:-1]+w[...,1:]))[...,::-1],
       's3d':xp.zeros_like(th),'dt':dt,'dtmn':dtmn,
       'sdtl':xp.asarray(r(0)),'dtl':xp.asarray(r(0))}
    for name in PROGNOSTICS:
        value=xp.asarray(fields[name],dtype=th.dtype)[...,::-1]
        if name in ('qdcn','qtcn','qccn','qrcn','qnin'):value=xp.maximum(c['rlimit'],value)
        v[name+'3d']=value
    for name in SURFACE+PRIVATE_PRECIP:
        v[name]=xp.asarray(fields[name],dtype=th.dtype) if name in ('rainnc','snownc','grapnc','hailnc') else xp.zeros(th.shape[:-1],dtype=th.dtype)
    v=micro(v,c,xp=xp,observer=observer)
    result={'th':v['tk3d'][...,::-1]/pii}
    result.update({name:xp.fmax(r(0),v[name+'3d'][...,::-1]) for name in PROGNOSTICS})
    result.update({name:v[name] for name in SURFACE+PRIVATE_PRECIP})
    names=('th',*PROGNOSTICS,*SURFACE) if output_names is None else output_names
    return {name:result[name] for name in names}


def micro(v,c,*,xp=np,observer=None):
    """Shared column-local body; native entry supplies top-down register tiles."""
    r=np.dtype(v['tk3d'].dtype).type
    dt,dtmn=v['dt'],v['dtmn']
    v=phases.initialize(v,xp=xp,c=c)
    v=phases.environment(v,xp=xp,c=c)
    if observer is not None:observer('environment',v)
    count=xp.asarray((dt-c['dt20s']*r(.01))/c['dt20s'],dtype=np.int32)+np.int32(1)
    def macro(state):
        i,v=state
        v=dict(v)
        v['dtl']=xp.minimum(c['dt20s'],dt-v['sdtl'])
        v=phases.reset_small(v,xp=xp,c=c)
        def small(v):return phases.advance_small(v,xp=xp,c=c,active=v['sdts']<v['dtl'])
        v=while_loop(lambda v:any_true(v['sdts']<v['dtl'],xp=xp),small,v,xp=xp)
        v=phases.after_small(v,xp=xp,c=c)
        v=phases.coalescence(v,xp=xp,c=c)
        v['sdtl']=v['sdtl']+v['dtl']
        return i+np.int32(1),v
    _,v=while_loop(lambda s:s[0]<count,macro,(np.int32(0),v),xp=xp)
    if observer is not None:observer('processes',v)
    v=phases.before_sedimentation(v,xp=xp,c=c)
    if observer is not None:observer('before_sedimentation',v)
    precips={'qc':'clodncv','qr':'rainncv','qi':'icencv','qs':'snowncv','qg':'grapncv','qh':'hailncv',
             'vi':'vincv','vs':'vsncv','vg':'vgncv','fi':'fincv','fs':'fsncv','ai':'aincv','as':'asncv','ag':'agncv','ah':'ahncv','i3m':'i3mncv'}
    for name,target in precips.items():
        velocity=v['vti3m' if name=='i3m' else 'vt_'+name]
        v[name+'3d'],v[target]=flux(v[name+'3d'],velocity,v['rho'],v['dz3d'],dt,dtmn,xp=xp)
    v['rainnc']=v['rainnc']+v['clodncv']+v['rainncv']+v['icencv']+v['snowncv']+v['grapncv']+v['hailncv']
    v['snownc']=v['snownc']+v['icencv']+v['snowncv']
    v['grapnc']=v['grapnc']+v['grapncv'];v['hailnc']=v['hailnc']+v['hailncv']
    v['sr']=(v['icencv']+v['snowncv']+v['grapncv']+v['hailncv'])/(v['clodncv']+v['rainncv']+v['icencv']+v['snowncv']+v['grapncv']+v['hailncv']+r(1.e-12))
    for name in ('nc','nr','ni','ns','ng','nh'):
        v[name+'3d'],_=flux(v[name+'3d'],v['vt_'+name],v['rho'],v['dz3d'],dt,xp=xp)
    # Original NAERN=(4,1): only the cloud/rain sulfate channels settle.
    aero=v['aero']
    from . import _native
    from .support import get_local,_set_local
    for plane,speed in ((2,0),(3,1)):
        if _native.active.get():
            updated,_=flux(get_local(aero,plane,xp=xp),get_local(v['vtmean'],speed,xp=xp),v['rho'],v['dz3d'],dt,xp=xp)
            aero=_set_local(aero,(plane,),updated,True,xp=xp)
        else:
            updated,_=flux(aero[...,plane],v['vtmean'][...,speed],v['rho'],v['dz3d'],dt,xp=xp)
            select=xp.arange(aero.shape[-1])==plane
            aero=xp.where(select,updated[...,None],aero)
    v['aero']=aero
    if observer is not None:observer('sedimentation',v)
    v=phases.finish(v,xp=xp,c=c)
    return v
