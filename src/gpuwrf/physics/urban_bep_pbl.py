"""WRF BEP/BEM PBL consumers. Called only with sf_urban_physics=2/3."""
import jax
import jax.numpy as jnp


def _invert(lower,diagonal,upper,rhs):
    """module_bl_boulac.F:invert; reverse elimination, then forward substitution."""
    n=rhs.shape[-1];diag=[diagonal[...,k] for k in range(n)];b=[rhs[...,k] for k in range(n)]
    for k in range(n-2,-1,-1):
        b[k]=b[k]-upper[...,k]*b[k+1]/diag[k+1]
        diag[k]=diag[k]-upper[...,k]*lower[...,k+1]/diag[k+1]
    for k in range(1,n):b[k]=b[k]-lower[...,k]*b[k-1]/diag[k-1]
    return jnp.stack([b[k]/diag[k] for k in range(n)],axis=-1)


def _diff(co,rho,rhoz,cd,aa,bb,surface,volume,dz,dt):
    c0=surface[0]*rhoz[0]*cd[0]/dz[0]
    ci=2.*surface[1:-1]*rhoz[1:-1]*cd[1:-1]/(dz[1:]+dz[:-1])
    ct=surface[-1]*rhoz[-1]*cd[-1]/dz[-1]
    cddz=jnp.concatenate((c0[None],ci,ct[None]))
    dzv=volume[:-1]*dz[:-1]
    lower=jnp.zeros_like(co).at[:-1].set(-cddz[:-2]*dt/dzv/rho[:-1])
    diag=jnp.ones_like(co).at[:-1].set(1+dt*(cddz[:-2]+cddz[1:-1])/dzv/rho[:-1]-aa[:-1]*dt)
    upper=jnp.zeros_like(co).at[:-1].set(-cddz[1:-1]*dt/dzv/rho[:-1])
    rhs=co.at[:-1].set(co[:-1]+bb[:-1]*dt)
    result=_invert(lower,diag,upper,rhs)
    flux=jnp.concatenate((jnp.zeros_like(co[:1]),-(cddz[1:-1]*(result[1:]-result[:-1]))/rho[1:]))
    return result,flux


def ysu_tridi2n(lower,diag_u,diag_v,upper,rhs_u,rhs_v):
    """Literal bl_ysu.F90:tridi2n: BOTH components use the U upper factors.

    With different BEP a_u/a_v this differs from two independent Thomas solves.
    The shared factors are WRF's actual implemented caller contract.
    """
    n=len(rhs_u);fk=1./diag_u[0]
    au=[fk*upper[0]];u=[fk*rhs_u[0]];v=[(1./diag_v[0])*rhs_v[0]]
    for k in range(1,n-1):
        fk=1./(diag_u[k]-lower[k]*au[k-1])
        au.append(fk*upper[k]);u.append(fk*(rhs_u[k]-lower[k]*u[k-1]))
    for k in range(1,n-1):
        fk=1./(diag_v[k]-lower[k]*au[k-1])
        v.append(fk*(rhs_v[k]-lower[k]*v[k-1]))
    fk=1./(diag_u[-1]-lower[-1]*au[-1]);u.append(fk*(rhs_u[-1]-lower[-1]*u[-1]))
    fk=1./(diag_v[-1]-lower[-1]*au[-1]);v.append(fk*(rhs_v[-1]-lower[-1]*v[-1]))
    for k in range(n-2,-1,-1):
        u[k]=u[k]-au[k]*u[k+1];v[k]=v[k]-au[k]*v[k+1]
    return jnp.stack(u),jnp.stack(v)


def boulac_urban_column(u,v,theta,qv,qc,rho,dz,tke,hfx,qfx,ust,dt,cp,g,urban):
    from gpuwrf.physics import pbl_boulac as b
    z=b._edge_heights(dz);rhoz=b._rhoz(rho,dz);frc=urban.fraction
    dlg=(z[:-1]+z[1:])/2.*(1.-frc)+urban.ground_length*frc
    dl_u=jnp.where(frc>0.,urban.building_length/jnp.where(frc>0.,frc,1.),urban.building_length)
    surface=urban.surface.at[-1].set(urban.surface[0]);volume=urban.volume
    pblh=b._pbl_height(dz,z,theta,qv)
    wts=jnp.maximum(0.,hfx/rho[0]/cp)
    wstar=(g*wts*pblh/300.)**(1./3.)
    face=jnp.arange(len(dz)+1)
    gamma=jnp.where((face>=1)&(face<=len(dz)-1)&(z<=pblh)&(wts>0.),10.*wts/wstar/pblh,0.)
    dlu,dld=b._dissip_bougeault(g,z,dz,tke,theta)
    dld=jnp.minimum(dld,dlg);dls=jnp.sqrt(dlu*dld);dlk=jnp.minimum(dlu,dld)
    safe=jnp.where(dl_u>0.,dl_u,1.)
    dls=jnp.where(dl_u>0.,1./(1./dls+1./safe),dls)
    dlk=jnp.where(dl_u>0.,1./(1./dlk+1./safe),dlk)
    exch=b._cdtur_bougeault(tke,dz,dlk)
    sh=b._shear_bougeault(u,v,dz,exch)*surface[:-1]
    bu=b._buoy_bougeault(theta,dz,exch,gamma,g)*surface[:-1]
    ceps=jnp.asarray(1.,theta.dtype)/jnp.asarray(1.4,theta.dtype)
    td=-ceps*jnp.sqrt(jnp.maximum(tke,1.e-4))/jnp.maximum(dls,.1)
    ae=urban.a_e+td;be=urban.b_e+sh+bu
    tke_new,_=_diff(tke,rho,rhoz,exch,ae,be,surface,volume,dz,dt)
    tke_new=jnp.maximum(tke_new,1.e-4)
    gamma_div=(exch[1:]*gamma[1:]-exch[:-1]*gamma[:-1])/dz
    active=(jnp.arange(len(dz))>=1)&(z[:-1]<=pblh)&(wts>0.)
    bt=urban.b_t-jnp.where(active,gamma_div,0.)
    u_new,wu=_diff(u,rho,rhoz,exch,urban.a_u,urban.b_u,surface,volume,dz,dt)
    v_new,wv=_diff(v,rho,rhoz,exch,urban.a_v,urban.b_v,surface,volume,dz,dt)
    th_new,wt=_diff(theta,rho,rhoz,exch,urban.a_t,bt,surface,volume,dz,dt)
    q_new,wq=_diff(qv,rho,rhoz,exch,urban.a_q,urban.b_q,surface,volume,dz,dt)
    qc_new,_=_diff(qc,rho,rhoz,exch,jnp.zeros_like(qc),jnp.zeros_like(qc),surface,volume,dz,dt)
    return ((u_new-u)/dt,(v_new-v)/dt,(th_new-theta)/dt,(q_new-qv)/dt,(qc_new-qc)/dt,
            tke_new,dlk,exch[:-1],exch[:-1],pblh,wu,wv,wt,wq,dlg,ae,be)


def boulac_urban_columns(u,v,theta,qv,qc,rho,dz,tke,*,hfx,qfx,ust,dt,cp,g,urban):
    n=len(u);dtype=u.dtype
    broadcast=lambda x:jnp.broadcast_to(jnp.asarray(x,dtype),(n,))
    out=jax.vmap(boulac_urban_column)(u,v,theta,qv,qc,rho,dz,tke,hfx,qfx,ust,
                                   broadcast(dt),broadcast(cp),broadcast(g),urban)
    keys=('u','v','theta','qv','qc','tke','dlk','exch_h','exch_m','pblh','wu','wv','wt','wq','urban_dlg','urban_ae','urban_be')
    return dict(zip(keys,out))


def ysu_urban_columns(u,v,temperature,qv,pressure,pressure_interface,exner,dz,*,
                     psfc,znt,ust,hfx,qfx,wspd,br,psim,psih,dt,xland,u10,v10,uoce,voce,urban,
                     qc=None,qi=None,rthraten=None,ysu_topdown_pblmix=1,ctopo=1.0,ctopo2=1.0):
    from gpuwrf.physics.pbl_ysu import _ysu_column_traceable,_YSU_OUT_KEYS
    n=len(u)
    if uoce is None:uoce=jnp.zeros(n,dtype=u.dtype)
    if voce is None:voce=jnp.zeros(n,dtype=u.dtype)
    dt_b=jnp.broadcast_to(jnp.asarray(dt,u.dtype),(n,))
    col=lambda x:jnp.broadcast_to(jnp.asarray(x,u.dtype),(n,))
    f_qc,f_qi=qc is not None,qi is not None
    zero=jnp.zeros_like(u);topdown=int(ysu_topdown_pblmix)==1
    def column(*a):
        return _ysu_column_traceable(*a[:8],psfc=a[8],znt=a[9],ust=a[10],hfx=a[11],qfx=a[12],
            wspd=a[13],br=a[14],psim=a[15],psih=a[16],dt=a[17],xland=a[18],u10=a[19],v10=a[20],
            uoce=a[21],voce=a[22],urban=a[23],qc=a[24] if f_qc else None,qi=a[25] if f_qi else None,
            rthraten=a[26],ctopo=a[27],ctopo2=a[28],topdown=topdown)
    out=jax.vmap(column)(u,v,temperature,qv,pressure,pressure_interface,exner,dz,
        psfc,znt,ust,hfx,qfx,wspd,br,psim,psih,dt_b,xland,u10,v10,uoce,voce,urban,
        qc if f_qc else zero,qi if f_qi else zero,zero if rthraten is None else rthraten,col(ctopo),col(ctopo2))
    result=dict(zip(_YSU_OUT_KEYS,out))
    if not f_qc:result.pop('qc')
    if not f_qi:result.pop('qi')
    return result
