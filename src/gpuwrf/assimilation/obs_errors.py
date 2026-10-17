"""WRF ``errob``: observation innovations on the C grid (root domain).

Inputs/outputs reproduce module_fddaobs_rtfdda.F's boundary. Model arrays are
(k,j,i); VAROBS/ERRF use (observation,variable). VAROBS columns are U,V,T(K),
QV(kg/kg),pressure(cb),height(m). RKO and the possibly modified VAROBS persist
between calls, as in WRF. Ingestion must provide finite values or numeric WRF
missing sentinels and coordinates admitted by in4dob (2 <= rio/rjo <= ide/jde-1).
"""

import jax.numpy as jnp
import numpy as np

_RCP_REAL = np.float32(np.float32(287.0) / np.float32(1004.5))


def _bilerp(field, i, j, dx, dy):
    one = jnp.asarray(1, field.dtype)
    return ((one-dy)*((one-dx)*field[...,j,i] + dx*field[...,j,i+1])
            + dy*((one-dx)*field[...,j+1,i] + dx*field[...,j+1,i+1]))


def _trilerp(field, i, j, k, kp, dx, dy, dz):
    one = jnp.asarray(1, field.dtype)
    low = ((one-dy)*((one-dx)*field[k,j,i] + dx*field[k,j,i+1])
           + dy*((one-dx)*field[k,j+1,i] + dx*field[k,j+1,i+1]))
    high = ((one-dy)*((one-dx)*field[kp,j,i] + dx*field[kp,j,i+1])
            + dy*((one-dx)*field[kp,j+1,i] + dx*field[kp,j+1,i+1]))
    return (one-dz)*low + dz*high


def errob(varobs, rko, rio, rjo, *, u, v, tb, qv, pbase, pp, z,
          uratx, vratx, tratx, kpbl, wind=True, temperature=True, moisture=True):
    """Return (modified VAROBS, RKO, ERRF) for one complete root-domain call.

    tb is WRF TH_PHY_M_T0 (dry potential temperature minus 300 K), not moist
    theta. uratx/vratx/tratx are surface_driver's current similarity ratios;
    kpbl is the actual PBL scheme's 1-based index. No ratios/PBL guesses here.
    """
    dtype = tb.dtype
    r = lambda x:jnp.asarray(x,dtype)  # noqa: E731
    nz,ny,nx = tb.shape
    rio,rjo,rko,varobs = map(r,(rio,rjo,rko,varobs))
    im0,jm0 = jnp.trunc(rio-r(.5)).astype(jnp.int32),jnp.trunc(rjo-r(.5)).astype(jnp.int32)
    dxm,dym = rio-r(.5)-r(im0),rjo-r(.5)-r(jm0)
    im,jm = jnp.clip(im0,1,nx)-1,jnp.clip(jm0,1,ny)-1
    pbbo = r(.001)*_bilerp(pbase,im,jm,dxm,dym)
    ppbo = r(.001)*_bilerp(pp,im,jm,dxm,dym)
    zcol = _bilerp(z,im,jm,dxm,dym)
    levels = jnp.arange(1,nz+1)[:,None]
    pob = varobs[:,4]
    need_location = jnp.abs(rko+r(99))<r(1)
    height_only = (pob==r(-888888)) & (varobs[:,5]>r(-800000))
    khi = jnp.min(jnp.where((levels>=2) & (varobs[:,5]<=zcol),levels,nz),axis=0)-1
    klo = khi-1
    n = jnp.arange(len(rko))
    zlo,zhi = zcol[klo,n],zcol[khi,n]
    ln_plo = jnp.log(pbbo[khi,n]+ppbo[khi,n])
    ln_phi = jnp.log(pbbo[klo,n]+ppbo[klo,n])
    ln_p = ln_plo+(ln_phi-ln_plo)*((zhi-varobs[:,5])/(zhi-zlo))
    ln_p = jnp.where(varobs[:,5]<=zlo,ln_phi,jnp.where(varobs[:,5]>=zhi,ln_plo,ln_p))
    pob = jnp.where(height_only & need_location,jnp.exp(ln_p),pob)
    # Original loop searches from k_end-1 downward, then extrapolates/clamps.
    kbot = jnp.max(jnp.where((levels<nz) & (pob<=pbbo+ppbo),levels,1),axis=0)-1
    pbhi,pphi = pbbo[kbot+1,n],ppbo[kbot+1,n]
    located = r(kbot+2)-(pob-pbhi-pphi)/(pbbo[kbot,n]+ppbo[kbot,n]-pbhi-pphi)
    rko = jnp.where(need_location & (pob>r(-800000)),jnp.maximum(located,r(1)),rko)
    varobs = varobs.at[:,4].set(jnp.where(height_only,jnp.where(need_location,pob,r(0)),varobs[:,4]))
    ic = jnp.trunc(rko).astype(jnp.int32)
    k = jnp.minimum(nz,ic)-1
    kp = jnp.minimum(k+1,nz-1)
    dz = rko-r(ic)

    # upoint/vpoint boundary extrapolation, exactly as the errob caller uses.
    uratio = jnp.concatenate((uratx[:,:1],r(.5)*(uratx[:,1:]+uratx[:,:-1]),uratx[:,-1:]),axis=1)
    vratio = jnp.concatenate((vratx[:1],r(.5)*(vratx[1:]+vratx[:-1]),vratx[-1:]),axis=0)
    err = jnp.zeros((len(rko),10),dtype)
    for ityp,(gx,gy) in enumerate(((0.,.5),(.5,0.),(.5,.5))):
        ra,rb = rio-r(gx),rjo-r(gy)
        ia,ib = jnp.trunc(ra).astype(jnp.int32),jnp.trunc(rb).astype(jnp.int32)
        dx,dy = ra-r(ia),rb-r(ib)
        dummy = (r(99999)*((ra<r(1)).astype(jnp.int32)+(rb<r(1)).astype(jnp.int32)
                 +((r(nx+1)-r(2*gx)-ra-r(1e-10))<r(0)).astype(jnp.int32)
                 +((r(ny+1)-r(2*gy)-rb-r(1e-10))<r(0)).astype(jnp.int32)
                 +(rko<r(1)).astype(jnp.int32)))
        i,j = jnp.clip(ia,2 if ityp<2 else 1,nx)-1,jnp.clip(ib,2 if ityp<2 else 1,ny)-1
        if ityp<2:
            ratio = _bilerp(uratio if ityp==0 else vratio,i,j,dx,dy)
            ratio = jnp.where(rko==r(1),ratio,r(1))
            ratio = jnp.where(jnp.abs(ratio)<r(.001),r(1),ratio)
            value = dummy+ratio*varobs[:,ityp]-_trilerp(u if ityp==0 else v,i,j,k,kp,dx,dy,dz)
            err = err.at[:,ityp].set(value if wind else dummy)
            if ityp==0:
                iminus,iplus = jnp.maximum(i-1,0),jnp.minimum(i+1,nx-1)
                p1 = r(.5)*(pbase[0,j,iminus]+pbase[0,j,i])
                p2 = r(.5)*(pbase[0,j,i]+pbase[0,j,iplus])
                ps = r(.001)*((r(1)-dx)*p1+dx*p2)
            else:
                jminus,jplus = jnp.maximum(j-1,0),jnp.minimum(j+1,ny-1)
                p1 = r(.5)*(pbase[0,jminus,i]+pbase[0,j,i])
                p2 = r(.5)*(pbase[0,j,i]+pbase[0,jplus,i])
                ps = r(.001)*((r(1)-dy)*p1+dy*p2)
            err = err.at[:,6+ityp].set(ps if wind else r(0))
        else:
            err = err.at[:,2:5].set(dummy[:,None])
            if temperature or moisture:
                tr = _bilerp(tratx,i,j,dx,dy)
                tr = jnp.where(rko==r(1),tr,r(1))
                tr = jnp.where(jnp.abs(tr)<r(.001),r(1),tr)
                cp = r(_RCP_REAL if dtype==jnp.float32 else 287.0/1004.5)
                obtheta = jnp.where((varobs[:,2]>r(-800000)) & (varobs[:,4]>r(-800000)),
                                    varobs[:,2]*(r(100)/varobs[:,4])**cp-r(300),r(-888888))
                err = err.at[:,2].set(dummy+tr*obtheta-_trilerp(tb,i,j,k,kp,dx,dy,dz))
                qm = _trilerp(qv,i,j,k,kp,dx,dy,dz)
                err = err.at[:,3].set(dummy+varobs[:,3]-qm)
                err = err.at[:,9].set(qm)
                err = err.at[:,5].set(r(.001)*_bilerp(pbase[0],i,j,dx,dy))
            nearest_i = i+jnp.floor(dx+r(.5)).astype(jnp.int32)
            nearest_j = j+jnp.floor(dy+r(.5)).astype(jnp.int32)
            err = err.at[:,4].set(kpbl[nearest_j,nearest_i])
    err = err.at[:,8].set(rko)
    gross = (jnp.abs(err[:,2])>r(20)) & (err[:,2]>r(-800000))
    err = err.at[:,:4].set(jnp.where(gross[:,None],r(-888888),err[:,:4]))
    varobs = varobs.at[:,:4].set(jnp.where(gross[:,None],r(-888888),varobs[:,:4]))
    return varobs,rko,err
