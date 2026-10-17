"""WRF SMS vertical column work; one Pallas launch per routine, REAL operands.

Source: module_diffusion_em.F nonlocal_flux/free_atmos_length/meso_length_scale.
Level/search loops execute inside the column kernel, never across the domain.
Arrays use (levels, columns), preserving level-major producer storage.
"""
from functools import partial
import math
import jax
from jax import lax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as pt
from gpuwrf.kernels.dyn_ring_relax_fp32 import _rn

def _nr(op,a,b):
    # Integer indexing remains exact; only WRF REAL arithmetic is rounded.
    if jnp.issubdtype(jnp.result_type(a,b),jnp.integer):
        return {"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"div":lambda:a/b}[op]()
    return _rn(op,a,b,interpret=jax.default_backend()=="cpu")


BLOCK = 128
F = jnp.float32


def _height_kernel(rdzw, z, *, nz, nc):
    col = pl.program_id(0)*BLOCK+jnp.arange(BLOCK, dtype=jnp.int32)
    valid = col < nc
    zz = jnp.zeros((BLOCK,), F)
    pt.store(z.at[0, col], zz, mask=valid)
    for k in range(nz):
        zz = F(1)/pt.load(rdzw.at[k,col], mask=valid, other=1)+zz
        pt.store(z.at[k+1,col], zz, mask=valid)


def _free_kernel(rdzw, theta, tke, z, out, *, nz, nc):
    lane=pl.program_id(0)*BLOCK+jnp.arange(BLOCK,dtype=jnp.int32)
    col=lane//nz; src=lane%nz; valid=col<nc
    load=lambda ref,k:pt.load(ref.at[k,col],mask=valid,other=1)
    zero=jnp.zeros((BLOCK,),F); beta=F(9.81)/F(300)
    ts=load(theta,src); energy=load(tke,src); dzs=F(1)/load(rdzw,src)
    zsrc=load(z,src); top=load(z,nz)
    def search(up):
        initial=top-zsrc-dzs/F(2) if up else zsrc+dzs/F(2)
        alive=valid & ((src<nz-1) if up else (src>0))
        def cond(s):return jnp.max(s[-1].astype(jnp.int32))>0
        def step(s):
            idx,pe,depth,distance,active=s
            safe=jnp.minimum(idx,nz-2) if up else jnp.maximum(idx,1)
            nxt=safe+1 if up else safe-1
            delta=(F(1)/load(rdzw,nxt)+F(1)/load(rdzw,safe))/F(2)
            t0,t1=load(theta,safe),load(theta,nxt)
            pe_new=pe-beta*ts*delta if up else pe+beta*ts*delta
            pe_new=pe_new+beta*(t1+t0)*delta/F(2) if up else pe_new-beta*(t1+t0)*delta/F(2)
            dd=depth+delta
            cross=active&(energy<pe_new)&(energy>=pe)
            slope=(t1-t0)/delta if up else (t0-t1)/delta
            q=beta*(t0-ts)
            rad=jnp.maximum(F(0),q*q+F(2)*slope*beta*(energy-pe))
            numerator=(-q if up else q)+jnp.sqrt(rad)
            quad=numerator/jnp.where(slope!=0,slope,F(1))/beta
            lin=(energy-pe)/(beta*jnp.where(t0!=ts,t0-ts,F(1)))
            tl=jnp.where(slope!=0,quad,jnp.where(t0!=ts,lin,F(0)))
            distance=jnp.where(cross,dd-delta+tl,distance)
            newidx=idx+1 if up else idx-1
            more=newidx<nz-1 if up else newidx>0
            return newidx,jnp.where(active,pe_new,pe),jnp.where(active,dd,depth),distance,active&~cross&more
        return lax.while_loop(cond,step,(src,zero,zero,initial,alive))[3]
    dlu=search(True); dld=jnp.minimum(search(False),(zsrc+load(z,src+1))/F(2))
    el=jnp.minimum(dlu,dld); el=el/(F(1)+el/F(2000))
    pt.store(out.at[src,col],el,mask=valid)


def _partial(d, h, nonlocal_heat=False, momentum=False):
    doh = _nr('div', d, jnp.where(h != 0, h, F(1)))
    if momentum:
        num = _nr('add', doh ** F(2), _nr('mul', F(0.07), doh ** F(0.6666667)))
        den = _nr('add', _nr('add', doh ** F(2), _nr('mul', F(0.142), doh ** F(0.6666667))), F(0.071))
        val = _nr('div', num, den)
    else:
        a2, a3, a5, a6, a7, exp = (0.936, -1.11, 0.312, 0.329, 0.243, 0.875) if nonlocal_heat else (0.87, -0.913, 0.153, 0.278, 0.28, 0.5)
        num = _nr('add', _nr('add', doh ** F(2), _nr('mul', F(a2), doh ** F(exp))), F(a3))
        den = _nr('add', _nr('add', doh ** F(2), _nr('mul', F(a5), doh ** F(exp))), F(a6))
        val = _nr('add', _nr('div', _nr('mul', F(a7), num), den), F(float(__import__('numpy').float32(1) - __import__('numpy').float32(a7))))
    return jnp.where(d <= F(100), F(0), jnp.minimum(jnp.maximum(jnp.where(h != 0, val, F(1)), F(0)), F(1)))


def _nonlocal_kernel(theta, qv, u, v, rho, z, ust, hfx, br, wspd, mtx, mty, oldgu, oldgv, dxref, dyref, nl, gu, gv, hp, probe=None, *, nz, nc, isfflx, heat):
    col = _nr('add', _nr('mul', pl.program_id(0), BLOCK), jnp.arange(BLOCK, dtype=jnp.int32))
    valid = col < nc
    ld = lambda r, k: pt.load(r.at[k, col], mask=valid, other=1)
    s = lambda r: pt.load(r.at[col], mask=valid, other=1)
    ep1 = F(float(__import__('numpy').float32(__import__('numpy').float32(461.6) / __import__('numpy').float32(287)) - __import__('numpy').float32(1)))
    tv = lambda k: _nr('mul', ld(theta, k), _nr('add', F(1), _nr('mul', ep1, ld(qv, k))))
    th0 = ld(theta, 0)
    tv0 = tv(0)
    cpm = _nr('mul', F(1004.5), _nr('add', F(1), _nr('mul', F(0.8), ld(qv, 0))))
    hf = s(hfx) if isfflx == 1 else _nr('mul', _nr('mul', F(heat), cpm), ld(rho, 0))
    sf = _nr('div', _nr('div', hf, cpm), ld(rho, 0))
    sfc = s(br) <= 0
    ustv = s(ust)
    height = jnp.zeros((BLOCK,), F)
    kp = jnp.ones((BLOCK,), jnp.int32)
    found = jnp.zeros((BLOCK,), jnp.bool_)
    prev = tv0
    for k in range(1, nz):
        th = tv(k)
        cross = (th > _nr('add', tv0, F(0.5))) & ~found
        za0 = _nr('div', _nr('add', ld(z, _nr('sub', k, 1)), ld(z, k)), F(2))
        za1 = _nr('div', _nr('add', ld(z, k), ld(z, _nr('add', k, 1))), F(2))
        hv = _nr('add', za0, _nr('mul', _nr('div', _nr('sub', _nr('add', tv0, F(0.5)), prev), jnp.maximum(F(0.01), _nr('sub', th, prev))), _nr('sub', za1, za0)))
        height = jnp.where(cross, hv, height)
        kp = jnp.where(cross, _nr('add', k, 1), kp)
        found = found | cross
        prev = th
    kp = jnp.where(height < ld(z, 1), 1, kp)
    w3 = jnp.where(sfc, _nr('mul', _nr('mul', _nr('div', F(9.81), th0), jnp.maximum(sf, F(0))), height), F(0))
    # WRF REAL 8*karman = REAL(3.2), byte-exact (8 is a power of two).
    # A scalar literal-only inline-asm op is unsupported by Triton lowering.
    scale = _nr('add', ustv ** 3, _nr('mul', _nr('mul', F(3.2), w3), F(0.5))) ** F(0.33333335)
    scale = jnp.maximum(jnp.minimum(scale, _nr('mul', ustv, F(16))), _nr('div', ustv, F(5)))
    dx,dy=pt.load(dxref.at[()]),pt.load(dyref.at[()])
    delxy = jnp.sqrt(_nr('div', _nr('mul', _nr('div', dx, s(mtx)), dy), s(mty)))
    pu = _partial(delxy, height, momentum=True)
    active = sfc & (sf > 0) & (height > 0)
    pbl = active & (kp > 1)
    hs = jnp.where(active, height, F(1))
    ws = jnp.where(active, scale, F(1))
    bv = _nr('div', _nr('mul', _nr('mul', _nr('mul', F(-10.9), ustv), ustv), w3), _nr('mul', hs, ws ** 4))
    wind = jnp.where(active, s(wspd), F(1))
    pt.store(gu.at[col], jnp.where(active, _nr('div', _nr('mul', _nr('mul', pu, bv), ld(u, 0)), wind), s(oldgu)), mask=valid)
    pt.store(gv.at[col], jnp.where(active, _nr('div', _nr('mul', _nr('mul', pu, bv), ld(v, 0)), wind), s(oldgv)), mask=valid)
    pt.store(hp.at[col], height, mask=valid)
    k = jnp.maximum(_nr('sub', kp, 2), 0)
    wm3 = _nr('add', w3, _nr('mul', F(5), ustv ** 3))
    wm2 = wm3 ** F(0.6666667)
    bflux = _nr('div', _nr('mul', _nr('div', _nr('mul', F(-0.15), tv0), F(9.81)), wm3), hs)
    dtv = jnp.maximum(_nr('sub', tv(_nr('add', k, 1)), tv(k)), F(0.01))
    dtheta = jnp.maximum(_nr('sub', ld(theta, _nr('add', k, 1)), ld(theta, k)), F(0.01))
    du, dv = (_nr('sub', ld(u, _nr('add', k, 1)), ld(u, k)), _nr('sub', ld(v, _nr('add', k, 1)), ld(v, k)))
    we = jnp.maximum(_nr('div', bflux, dtv), -jnp.sqrt(wm2))
    za = _nr('div', _nr('add', ld(z, k), ld(z, _nr('add', k, 1))), F(2))
    delta = _nr('add', _nr('mul', F(0.02), height), _nr('div', _nr('mul', F(0.05), wm2), _nr('mul', _nr('div', F(9.81), th0), dtv)))
    delta = jnp.minimum(jnp.maximum(_nr('mul', F(1.5), delta), _nr('sub', _nr('sub', height, za), F(1))), height)
    shear = _nr('add', du ** F(2), dv ** F(2))
    rigs = jnp.where(shear != 0, _nr('div', _nr('mul', _nr('mul', _nr('div', F(9.81), th0), dtv), delta), jnp.where(shear != 0, shear, F(1))), F(100))
    rigs = jnp.maximum(jnp.minimum(rigs, F(100)), F(-100))
    rr = jnp.where(pbl, rigs, F(100))
    w3safe = jnp.where(pbl, w3, F(1))
    enfr = _nr('mul', jnp.maximum(jnp.minimum(_nr('div', _nr('div', wm3, w3safe), _nr('add', F(1), _nr('div', F(-0.4), rr))), F(5)), F(1)), F(-0.15))
    delta_abs = jnp.where(pbl, delta, F(1))
    delt = _nr('div', delta_abs, jnp.where(pbl, height, F(1)))
    ml = _nr('sub', F(1), delt)
    sfcfrac = jnp.maximum(F(0.075), jnp.minimum(jnp.maximum(_nr('div', ld(z, 1), hs), F(1e-08)), F(1)))
    snl = _nr('mul', F(0.68), _nr('mul', _nr('add', F(1), _nr('mul', F(-1.15), sfcfrac)), sf))
    a1 = _nr('div', snl, sfcfrac)
    a2 = _nr('div', -snl, _nr('sub', ml, sfcfrac))
    b2 = _nr('mul', -ml, a2)
    a3 = _nr('div', _nr('mul', snl, enfr), delt)
    b3 = _nr('mul', -a3, ml)
    pth = _partial(delxy, height, nonlocal_heat=True)
    hfp = _nr('mul', _nr('add', a3, b3), pth)
    if probe is not None:
        data = (height, kp.astype(F), _nr('div', F(9.81), th0), sf, w3, scale, delt, rigs, enfr, wm2, dtv, hfp, delxy, ml, sfcfrac, _nr('mul', _nr('add', F(1), _nr('mul', F(-1.15), sfcfrac)), sf), snl, a1, a2, b2, a3, b3, pth)
        for row, value in enumerate(data):
            pt.store(probe.at[row, col], value, mask=valid)
    for k in range(nz):
        zz = ld(z, _nr('add', k, 1))
        zfac = jnp.maximum(_nr('div', zz, hs), F(1e-08))
        ent = jnp.sqrt(_nr('div', _nr('sub', zz, height), delta_abs) ** 2)
        val = jnp.where(zfac <= sfcfrac, _nr('mul', a1, zfac), jnp.where(zfac <= ml, _nr('add', _nr('mul', a2, zfac), b2), F(0)))
        val = _nr('mul', _nr('add', val, _nr('mul', hfp, jnp.exp(-ent))), pth)
        pt.store(nl.at[k, col], jnp.where(pbl & (k < _nr('sub', kp, 1)), val, F(0)), mask=valid)
    pt.store(nl.at[nz, col], jnp.zeros((BLOCK,), F), mask=valid)


def _meso_kernel(tke,theta,rdz,rdzw,p8w,t8w,z,elmin,rmol,hp,rho,qv,hfx,out,hout,
                 *,nz,nc,isfflx,heat):
    col=pl.program_id(0)*BLOCK+jnp.arange(BLOCK,dtype=jnp.int32);valid=col<nc
    ld=lambda r,k:pt.load(r.at[k,col],mask=valid,other=1)
    s=lambda r:pt.load(r.at[col],mask=valid,other=1)
    elt=jnp.full((BLOCK,),F(1.e-5));vsc=jnp.full((BLOCK,),F(1.e-5))
    for k in range(nz):
        qdz=jnp.sqrt(F(2)*ld(tke,k))*(F(1)/ld(rdzw,k))
        za=(ld(z,k)+ld(z,k+1))/F(2);elt=elt+qdz*za;vsc=vsc+qdz
    elt=F(.8)*elt/vsc
    cpm=F(1004.5)*(F(1)+F(.8)*ld(qv,0));hf=s(hfx) if isfflx==1 else F(heat)*cpm*ld(rho,0)
    sf=(hf/cpm)/ld(rho,0);gtr=F(9.81)/F(300)
    qcv=(gtr*jnp.maximum(sf,F(0))*elt)**F(float(__import__('numpy').float32(1)/__import__('numpy').float32(3)))
    zi=jnp.maximum(s(hp),F(300));h1=jnp.minimum(jnp.maximum(F(.3)*zi,F(300)),F(750));h2=h1/F(2)
    for k in range(nz):
        if k==0:
            tmpdz=_nr('add',_nr('div',F(1),ld(rdzw,1)),_nr('div',F(1),ld(rdzw,0)))
            ths=_nr('div',ld(t8w,0),_nr('div',ld(p8w,0),F(100000))**F(float(__import__('numpy').float32(287)/__import__('numpy').float32(1004.5))))
            dd=_nr('div',_nr('sub',ld(theta,1),ths),tmpdz)
        elif k==nz-1:
            tmpdz=_nr('add',_nr('div',F(1),ld(rdz,k)),_nr('div',F(.5),ld(rdzw,k)))
            tht=_nr('div',ld(t8w,nz),_nr('div',ld(p8w,nz),F(100000))**F(float(__import__('numpy').float32(287)/__import__('numpy').float32(1004.5))))
            dd=_nr('div',_nr('sub',tht,ld(theta,k-1)),tmpdz)
        else:
            tmpdz=_nr('add',_nr('div',F(1),ld(rdz,k+1)),_nr('div',F(1),ld(rdz,k)))
            dd=_nr('div',_nr('sub',ld(theta,k+1),ld(theta,k-1)),tmpdz)
        za=(ld(z,k)+ld(z,k+1))/F(2);qt=jnp.sqrt(F(2)*ld(tke,k))
        nn=jnp.sqrt(jnp.where(dd>0,gtr*dd,F(1)))
        elb=jnp.where(dd>0,qt/nn*(F(1)+F(1)*jnp.sqrt(qcv/(elt*nn))),F(1.e10))
        elf=jnp.where(dd>0,F(1)*qt/nn,elb)
        rs=s(rmol)
        els=jnp.where(rs>0,F(.4)*za/(F(1)+F(2.7)*jnp.minimum(za*rs,F(1))),F(1)*F(.4)*za*(F(1)-F(100)*za*jnp.minimum(rs,F(0)))**F(.2))
        dl=jnp.minimum(elb/(elb/elt+elb/els+F(1)),elf)
        wt=F(.5)*jnp.tanh((za-(zi+h1))/h2)+F(.5)
        dl=dl*(F(1)-wt)+F(.4)*ld(elmin,k)*wt
        pt.store(out.at[k,col],dl,mask=valid)
    pt.store(out.at[nz,col],jnp.zeros((BLOCK,),F),mask=valid)
    pt.store(hout.at[col],hf,mask=valid)


def _call(kernel,args,outputs,nc,name,**static):
    return pl.pallas_call(partial(kernel,nc=nc,**static),grid=((nc+BLOCK-1)//BLOCK,),
        out_shape=outputs,interpret=jax.default_backend()=='cpu',compiler_params=pt.CompilerParams(num_warps=4),name=name)(*args)


def upstream(rdzw,rdz,theta,p8w,t8w,tke,qv,rho,u,v,ust,hfx,br,rmol,wspd,msftx,msfty,gamu,gamv,*,dx,dy,isfflx,heat):
    """Input/output full interior columns; upstream height, nonlocal and lengths."""
    nz,ny,nx=tke.shape;nc=ny*nx
    prof=lambda x:jnp.asarray(x,F).reshape(x.shape[0],nc)
    surf=lambda x:jnp.asarray(x,F).reshape(nc)
    profile_out=jax.ShapeDtypeStruct((nz+1,nc),F);surface_out=jax.ShapeDtypeStruct((nc,),F)
    rz=prof(rdzw);th=prof(theta);tk=prof(tke)
    z=_call(_height_kernel,(rz,),profile_out,nc,'sms_heights',nz=nz)
    count=nz*nc
    el=pl.pallas_call(partial(_free_kernel,nz=nz,nc=nc),grid=((count+BLOCK-1)//BLOCK,),
        out_shape=jax.ShapeDtypeStruct((nz,nc),F),interpret=jax.default_backend()=='cpu',compiler_params=pt.CompilerParams(num_warps=4),name='sms_free_atmos')(rz,th,tk,z)
    nl,gu,gv,hp=_call(_nonlocal_kernel,(th,prof(qv),prof(u),prof(v),prof(rho),z,surf(ust),surf(hfx),surf(br),surf(wspd),surf(msftx),surf(msfty),surf(gamu),surf(gamv),jnp.asarray(dx,F),jnp.asarray(dy,F)),
        (profile_out,surface_out,surface_out,surface_out),nc,'sms_nonlocal_flux',nz=nz,isfflx=isfflx,heat=heat)
    dl,hout=_call(_meso_kernel,(tk,th,prof(rdz),rz,prof(p8w),prof(t8w),z,el,surf(rmol),hp,prof(rho),prof(qv),surf(hfx)),
        (profile_out,surface_out),nc,'sms_meso_length',nz=nz,isfflx=isfflx,heat=heat)
    return nl.reshape(nz+1,ny,nx),gu.reshape(ny,nx),gv.reshape(ny,nx),hp.reshape(ny,nx),dl.reshape(nz+1,ny,nx),el.reshape(nz,ny,nx),hout.reshape(ny,nx)
