"""Original2735..2835 adaptive fallout, column-local split counts and REAL order."""
import jax
import jax.numpy as jnp
from .math import _real
from . import _native
from .level_support import _get


def sedimentation(fields,*,xp=jnp):
    f=fields;_,r=_real(f['qr'],xp)
    rho=f['rhoair'];inverse=f['i_rhoair'];dz=f['dzmic'];dt=r(f['dt'])
    sedi=xp.asarray(f['sedi'],dtype=bool)
    maxfall=xp.maximum(xp.max(f['vtrmi1'],axis=-1),xp.max(f['vtrni1'],axis=-1))
    maxfall=xp.maximum(xp.maximum(maxfall,f['vtrm']),f['vtrn'])
    count=xp.max((maxfall*dt[...,None]/dz+r(1.)).astype('int32'),axis=-1)
    count=xp.where(sedi,xp.maximum(count,1),1)
    maximum=xp.max(xp.where(sedi,count,0))
    names=('qi','ni','ai','ci','qr','nr')
    speeds=(f['vtrmi1'],f['vtrni1'],f['vtrmi1'],f['vtrmi1'],f['vtrm'],f['vtrn'])
    old=[]
    for name in names:
        category=name in ('qi','ni','ai','ci')
        density=rho[...,None] if category else rho
        active=sedi[...,None,None] if category else sedi[...,None]
        old.append(xp.where(active,f[name]*density,f[name]))
    carry=(*old,f['qipre1d'],f['qrpre1d'])

    def step(nn,state):
        active=(nn<count)&sedi
        new=[];surface=[]
        for name,a,v in zip(names,state[:6],speeds):
            if _native.active.get():
                from gpuwrf.kernels.phys_thompson_sedimentation import _gather
                level=jnp.arange(rho.shape[-1],dtype=jnp.int32)
                top=level==_native.levels.get()-1
                def plane(old,velocity):
                    flux=velocity*old
                    above=_gather.bind(flux,(level+1)%rho.shape[-1])
                    above=jnp.where(level+1<_native.levels.get(),above,r(0.))
                    body=old+((above-flux)/dz)*dt/r(count)
                    last=old-(flux/dz)*dt/r(count)
                    updated=jnp.where(active,jnp.where(top,last,body),old)
                    return updated,jnp.sum(jnp.where(level==0,flux,jnp.zeros_like(flux)))
                if name in ('qi','ni','ai','ci'):
                    parts=[plane(_get(a,(i,)),_get(v,(i,))) for i in range(4)]
                    new.append(_native.stack4(tuple(x[0] for x in parts)))
                    surface.append(_native.stack4(tuple(x[1] for x in parts)))
                else:
                    updated,flux=plane(a,v);new.append(updated);surface.append(flux)
                continue
            flux=v*a
            if name in ('qi','ni','ai','ci'):
                body=a[...,:-1,:]+((flux[...,1:,:]-flux[...,:-1,:])/dz[...,:-1,None])*dt[...,None,None]/r(count)[...,None,None]
                top=a[...,-1:,:]-(flux[...,-1:,:]/dz[...,-1:,None])*dt[...,None,None]/r(count)[...,None,None]
                updated=xp.concatenate((body,top),axis=-2)
                updated=xp.where(active[...,None,None],updated,a)
                surface.append(flux[...,0,:])
            else:
                body=a[...,:-1]+((flux[...,1:]-flux[...,:-1])/dz[...,:-1])*dt[...,None]/r(count)[...,None]
                top=a[...,-1:]-(flux[...,-1:]/dz[...,-1:])*dt[...,None]/r(count)[...,None]
                updated=xp.concatenate((body,top),axis=-1)
                updated=xp.where(active[...,None],updated,a)
                surface.append(flux[...,0])
            new.append(updated)
        ice,rain=state[6:]
        # Original precipitation statements are in the interior k loop;
        # a one-level column never enters that loop. Preserve this behavior.
        if rho.shape[-1]>1:
            if _native.active.get():
                level=jnp.arange(rho.shape[-1],dtype=jnp.int32)
                ice_at_zero=jnp.sum(jnp.where((level==0)[...,None],new[0],jnp.zeros_like(new[0])),axis=-2)
                rain_at_zero=jnp.sum(jnp.where(level==0,new[4],jnp.zeros_like(new[4])))
            else:ice_at_zero=new[0][...,0,:];rain_at_zero=new[4][...,0]
            ice=xp.where(active[...,None]&(ice_at_zero>r(1.e-9)),
                         ice+surface[0]*dt[...,None]/r(count)[...,None],ice)
            rain=xp.where(active&(rain_at_zero>r(1.e-9)),
                          rain+surface[4]*dt/r(count),rain)
        return (*new,ice,rain)

    if xp is jnp:
        def body(state):
            nn,values=state
            return nn+1,step(nn,values)
        _,carry=jax.lax.while_loop(lambda state:state[0]<maximum,body,(jnp.int32(0),carry))
    else:
        for nn in range(int(maximum)):carry=step(nn,carry)
    result=dict(f)
    for name,value in zip(names,carry[:6]):
        category=name in ('qi','ni','ai','ci')
        factor=inverse[...,None] if category else inverse
        active=sedi[...,None,None] if category else sedi[...,None]
        result[name]=xp.where(active,value*factor,f[name])
    result.update(qipre1d=carry[6],qrpre1d=carry[7],sedimentation_steps=count)
    return result
