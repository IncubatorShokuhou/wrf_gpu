"""Resident pristine-WRF NWP diagnostics and pre-RK updraft helicity.

The two calls deliberately remain separate: ``cal_helicity`` precedes RK,
whereas ``diagnostic_output_nwp`` resets its maxima *after* RK at step 1 of
each history interval. Thompson's diagnostics caller does not supply Ng;
hail uses the diagnostic number distribution in module_diag_nwp.F.

Column recurrences and hail's DOUBLE bin sum run inside Pallas programs.
There is no bins x levels x domain temporary or host observation. CPU
interpretation is an operator-validation path, not a GPU performance claim.
"""
from functools import partial
import math
from typing import NamedTuple
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.kernels.dyn_rk_fp32 import _rn

BLOCK = 32
# Constant-only WRF REAL expressions are rounded before entering the Pallas
# jaxpr. Triton's inline-asm rule requires SSA operands, not scalar literals.
_AM_G = np.float32(np.float32(np.float32(3.1415926536)/np.float32(6))*np.float32(500))
_AM_G_GAMMA_RATIO = np.float32(np.float32(_AM_G*np.float32(6))/np.float32(1))
_THIRD = np.float32(np.float32(1)/np.float32(3))
_TWO_SEVENTHS = np.float32(np.float32(2)/np.float32(7))


class NwpDiagnostics(NamedTuple):
    wspd10max: object
    w_up_max: object
    w_dn_max: object
    up_heli_max: object
    w_mean: object
    grpl_max: object
    refd_max: object
    hail_maxk1: object
    hail_max2d: object
    # UH is a restart field; its unsmoothed physical boundary is retained.
    uh: object


def seed_nwp(state):
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    return NwpDiagnostics(*(zero for _ in NwpDiagnostics._fields))


def history_fields(diagnostics):
    return {} if diagnostics is None else {
        name.upper(): value for name, value in zip(diagnostics._fields[:-1], diagnostics[:-1])
    }


def validate_restart_nwp(diagnostics, enabled, version, shape=None):
    if diagnostics is None and not enabled:
        return
    if version != 1 or not isinstance(diagnostics, NwpDiagnostics):
        raise ValueError("restart NWP diagnostics schema is missing (E78)")
    if any(x is None or getattr(x, "dtype", None) != jnp.float32 for x in diagnostics):
        raise ValueError("restart NWP diagnostics lack REAL leaves (E78)")
    shapes = {x.shape for x in diagnostics}
    if len(shapes) != 1 or len(diagnostics.uh.shape) != 2:
        raise ValueError("restart NWP diagnostics have inconsistent shapes (E78)")
    if shape is not None and diagnostics.uh.shape != tuple(shape):
        raise ValueError("restart NWP diagnostics differ from the surface grid (E78)")


def _real_ops(interpret):
    def rn(op, a, b):
        a, b = jnp.broadcast_arrays(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32))
        return _rn(op, a, b, interpret=interpret)
    return tuple(partial(rn, op) for op in ("add", "sub", "mul", "div"))


def _hail_bins():
    # Literal DOUBLE initialization from pristine module_diag_nwp.F.
    edges = [.0005] + [math.exp(n / 50. * math.log(.075 / .0005) + math.log(.0005))
                           for n in range(1, 50)] + [.075]
    return tuple(math.sqrt(a*b) for a,b in zip(edges[:-1],edges[1:])), tuple(
        b-a for a,b in zip(edges[:-1],edges[1:]))


def _nwp_kernel(*refs, nz, ny, nx, supplied_ng, interpret):
    old, data, out = refs[:10], refs[10:-10], refs[-10:]
    u10,v10,w,p8w,znw,qg,rho,ph,phb,refl,ng,clock,bins,widths = data
    lane = pl.program_id(0)*BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
    y,x = lane//nx, lane%nx
    valid = lane < ny*nx
    add,sub,mul,div = _real_ops(interpret)
    r, d = jnp.float32, jnp.float64
    load = lambda a,k: pt.load(a.at[k,y,x],mask=valid,other=r(0))
    surface = lambda a: pt.load(a.at[y,x],mask=valid,other=r(0))
    step = pt.load(clock.at[0]).astype(jnp.int32)
    idump = pt.load(clock.at[1]).astype(jnp.int32)
    reset, normalize = (step-1)%idump == 0, step%idump == 0
    carried = tuple(jnp.where(reset,r(0),surface(a)) for a in old[:9])
    wind,up,dn,heli,mean,grpl,refd,hail1,hail2 = carried
    wind = jnp.maximum(wind,jnp.sqrt(add(mul(surface(u10),surface(u10)),mul(surface(v10),surface(v10)))))

    def level(k, s):
        up,dn,w_sum,count,grpl_sum,hail1,hail2 = s
        wk,pk = load(w,k),load(p8w,k)
        up = jnp.where((pk > r(40000)) & (wk > up),wk,up)
        dn = jnp.where((pk > r(40000)) & (wk < dn),wk,dn)
        eta = pt.load(znw.at[k])
        include = (eta >= r(.5)) & (eta <= r(.8))
        w_sum = jnp.where(include,add(w_sum,wk),w_sum)
        count = jnp.where(include,add(count,r(1)),count)

        def mass_level(t):
            grpl_sum,hail1,hail2 = t
            density = load(rho,k)
            q = load(qg,k)
            depth = sub(div(add(load(ph,k+1),load(phb,k+1)),r(9.81)),
                        div(add(load(ph,k),load(phb,k)),r(9.81)))
            grpl_sum = add(grpl_sum,mul(mul(q,depth),density))
            tqg = jnp.maximum(r(1.e-12),mul(q,density))
            active = valid & (tqg >= r(1.e-6))
            def hail_column(pair):
                hail1,hail2 = pair
                # xam_g is REAL, and all constants/WRF WGAMMA results here are REAL.
                am = r(_AM_G)
                if supplied_ng:
                    tng = jnp.maximum(r(1.e-8),mul(load(ng,k),density))
                else:
                    ygra = jnp.log10(jnp.maximum(r(1.e-9),tqg))
                    zans = add(r(3),mul(r(_TWO_SEVENTHS),add(ygra,r(8))))
                    zans = jnp.maximum(r(2),jnp.minimum(zans,r(6)))
                    # N0exp receives the result of a REAL power.
                    intercept = (r(10)**zans).astype(d)
                    lam_exp = (intercept*am.astype(d)*d(6)/tqg.astype(d))**d(.25)
                    # cgg(3)/cgg(2)/cgg(1) is REAL 1 for mu=0,b=3.
                    lamg = lam_exp
                    n0 = intercept/(d(1)*lam_exp)*lamg**d(1)
                    tng = jnp.maximum(d(np.float32(1.e-8)),n0*d(1)*lamg**d(-1)).astype(r)
                # WRF evaluates this parenthesis in REAL before DOUBLE power.
                base = div(mul(r(_AM_G_GAMMA_RATIO),tng),tqg)
                lamg = base.astype(d)**d(_THIRD)
                n0 = div(tng,r(1)).astype(d)*lamg**d(1)
                zeros = jnp.zeros((BLOCK,),d)
                ints = jnp.full((BLOCK,),49,jnp.int32)
                def bin_cond(s):
                    return jnp.max(s[-1].astype(jnp.int32)) > jnp.int32(0)
                def bin_step(s):
                    index,total,previous,alive = s
                    diameter = pt.load(bins.at[jnp.maximum(index,0)])
                    width = pt.load(widths.at[jnp.maximum(index,0)])
                    addition = n0*jnp.exp(-lamg*diameter)*width
                    updated = total+addition
                    cross = updated > d(.0005)
                    return (jnp.where(alive & ~cross,index-1,index),
                            jnp.where(alive,updated,total),
                            jnp.where(alive & ~cross,updated,previous),
                            alive & ~cross & (index > jnp.int32(0)))
                index,total,previous,_ = lax.while_loop(bin_cond,bin_step,(ints,zeros,zeros,active))
                diameter = pt.load(bins.at[jnp.maximum(index,0)])
                next_diameter = pt.load(bins.at[jnp.minimum(index+1,49)])
                interpolated = diameter-(total-d(.0005))/(total-previous)*(diameter-next_diameter)
                hail = jnp.where(index >= 49,pt.load(bins.at[49]),
                         jnp.where(next_diameter > r(1.e-3).astype(d),interpolated,d(np.float32(1.e-4)))).astype(r)
                hail1 = jnp.where(active & (k == 0),jnp.maximum(hail1,hail),hail1)
                hail2 = jnp.where(active,jnp.maximum(hail2,hail),hail2)
                return hail1,hail2
            # Dry tiles skip DOUBLE distribution work altogether; active
            # lanes retain WRF's bin-exit state independently within a tile.
            hail1,hail2 = lax.cond(jnp.max(active.astype(jnp.int32)) > jnp.int32(0),
                                  hail_column,lambda pair:pair,(hail1,hail2))
            return grpl_sum,hail1,hail2

        grpl_sum,hail1,hail2 = lax.cond(k < nz,mass_level,lambda t:t,(grpl_sum,hail1,hail2))
        return up,dn,w_sum,count,grpl_sum,hail1,hail2

    zero = jnp.zeros((BLOCK,),r)
    up,dn,w_sum,count,grpl_sum,hail1,hail2 = lax.fori_loop(0,nz+1,level,(up,dn,zero,zero,zero,hail1,hail2))
    mean = add(mean,div(w_sum,count))
    mean = jnp.where(normalize,div(mean,idump.astype(r)),mean)
    grpl = jnp.maximum(grpl,grpl_sum)
    refd = jnp.maximum(refd,load(refl,0))
    values = wind,up,dn,heli,mean,grpl,refd,hail1,hail2,surface(old[-1])
    for target,value in zip(out,values):
        pt.store(target.at[y,x],value,mask=valid)


def update_nwp(diagnostics, *, u10,v10,w,p8w,znw,qg,rho,ph,phb,refl_10cm,
               itimestep,dt,history_interval,ng=None,interpret=None):
    """WRF fixed-dt, Thompson NWP update on mass-grid owned cells.

    ``p8w/rho`` must be the diagnostics-driver held physics operands, not
    recomputed at output. ``itimestep`` is one-based, as in solve_em.F.
    Optional ``ng`` reproduces the operator's PRESENT branch; WRF's mp=8/28
    caller deliberately leaves it absent. Adaptive timesteps are unsupported.
    """
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    nz,ny,nx = qg.shape
    if float(history_interval) <= 0 or float(dt) <= 0:
        raise ValueError("NWP diagnostics require positive dt/history_interval")
    # WRF's assignment truncates a REAL division to an INTEGER.
    idump = int(np.float32(np.float32(history_interval)*np.float32(60))/np.float32(dt))
    if idump < 1:
        raise ValueError("NWP history interval must contain at least one timestep")
    if w.shape != (nz+1,ny,nx) or p8w.shape != w.shape or ph.shape != w.shape or phb.shape != w.shape:
        raise ValueError("NWP W/P8W/PH/PHB must be full z-face fields")
    bins,widths = _hail_bins()
    data = tuple(jnp.asarray(a,jnp.float32) for a in
                  (u10,v10,w,p8w,znw,qg,rho,ph,phb,refl_10cm,ng if ng is not None else qg))
    clock = jnp.stack((jnp.asarray(itimestep,jnp.int32),jnp.asarray(idump,jnp.int32)))
    shape = jax.ShapeDtypeStruct((ny,nx),jnp.float32)
    values = pl.pallas_call(partial(_nwp_kernel,nz=nz,ny=ny,nx=nx,supplied_ng=ng is not None,interpret=interpret),
                out_shape=(shape,)*10,grid=((ny*nx+BLOCK-1)//BLOCK,),interpret=interpret,
                name="nwp_thompson_column",compiler_params=pt.CompilerParams(num_warps=4))(
                *diagnostics,*data,clock,jnp.asarray(bins,jnp.float64),jnp.asarray(widths,jnp.float64))
    return NwpDiagnostics(*values)


def helicity_metrics(ph,phb,*,rdx,rdy):
    """Exact split-geopotential association of compute_diff_metrics (REAL)."""
    ph,phb = (jnp.asarray(a,jnp.float32) for a in (ph,phb))
    g,rdx,rdy = jnp.float32(9.81),jnp.float32(rdx),jnp.float32(rdy)
    z = (ph+phb)/g
    rdzw = jnp.float32(1)/(z[1:]-z[:-1])
    zx,zy = jnp.zeros_like(ph),jnp.zeros_like(ph)
    zx = zx.at[:,:,1:].set(rdx*(phb[:,:,1:]-phb[:,:,:-1])/g+rdx*(ph[:,:,1:]-ph[:,:,:-1])/g)
    zy = zy.at[:,1:,:].set(rdy*(phb[:,1:,:]-phb[:,:-1,:])/g+rdy*(ph[:,1:,:]-ph[:,:-1,:])/g)
    return zx,zy,rdzw


def _helicity_kernel(old_uh,u,v,w,ph,phb,msfux,msfvy,ht,rdzw,zx,zy,
                     fnm,fnp,dn,dnw,constants,uh_out,use_out,*,nz,ny,nx,interpret):
    lane = pl.program_id(0)*BLOCK+jnp.arange(BLOCK,dtype=jnp.int32)
    y,x = lane//nx,lane%nx
    valid = lane < ny*nx
    active = valid & (x >= 1) & (y >= 1)
    add,sub,mul,div = _real_ops(interpret)
    r = jnp.float32
    def l3(ref,k,yy=y,xx=x):
        return pt.load(ref.at[k,yy,xx],mask=active,other=r(0))
    def l2(ref,yy=y,xx=x):
        return pt.load(ref.at[yy,xx],mask=active,other=r(1))
    def vec(ref,k):
        return jnp.broadcast_to(pt.load(ref.at[k]),(BLOCK,))
    rdx,rdy,cf1,cf2,cf3 = (vec(constants,k) for k in range(5))
    cft2 = div(mul(r(-.5),vec(dnw,nz-1)),vec(dn,nz-1))
    cft1 = sub(r(1),cft2)
    mm = mul(mul(r(.25),add(l2(msfux,y-1,x),l2(msfux))),
                         add(l2(msfvy,y,x-1),l2(msfvy)))

    def hat(axis,k,left=False):
        if axis == "v":
            xx = x-1 if left else x
            return div(l3(v,k,y,xx),l2(msfvy,y,xx))
        yy = y-1 if left else y
        return div(l3(u,k,yy,x),l2(msfux,yy,x))

    def face(axis,k):
        def bottom(_):
            value = add(mul(cf1,hat(axis,0,True)),mul(cf2,hat(axis,1,True)))
            value = add(value,mul(cf3,hat(axis,2,True)))
            value = add(value,mul(cf1,hat(axis,0)))
            value = add(value,mul(cf2,hat(axis,1)))
            return mul(r(.5),add(value,mul(cf3,hat(axis,2))))
        def top(_):
            return mul(r(.5),add(mul(cft1,add(hat(axis,nz-1),hat(axis,nz-1,True))),
                                mul(cft2,add(hat(axis,nz-2),hat(axis,nz-2,True)))))
        def interior(_):
            kk = jnp.clip(k,1,nz-1)
            return mul(r(.5),add(mul(vec(fnm,kk),add(hat(axis,kk,True),hat(axis,kk))),
                                mul(vec(fnp,kk),add(hat(axis,kk-1,True),hat(axis,kk-1)))))
        return lax.cond(k == 0,bottom,lambda _:lax.cond(k == nz,top,interior,None),None)

    def sum4(a,b,c,d):
        return add(add(add(a,b),c),d)

    def vort(k):
        kk = jnp.minimum(k,nz-1)
        sx = mul(r(.25),sum4(l3(zx,kk,y-1,x),l3(zx,kk),l3(zx,kk+1,y-1,x),l3(zx,kk+1)))
        derivative = sub(face("v",kk+1),face("v",kk))
        invdz = sum4(l3(rdzw,kk),l3(rdzw,kk,y-1,x),l3(rdzw,kk,y-1,x-1),l3(rdzw,kk,y,x-1))
        correction = mul(mul(mul(derivative,r(.25)),sx),invdz)
        value = mul(mm,sub(mul(rdx,sub(hat("v",kk),hat("v",kk,True))),correction))
        sy = mul(r(.25),sum4(l3(zy,kk,y,x-1),l3(zy,kk),l3(zy,kk+1,y,x-1),l3(zy,kk+1)))
        derivative = sub(face("u",kk+1),face("u",kk))
        invdz = sum4(l3(rdzw,kk),l3(rdzw,kk,y,x-1),l3(rdzw,kk,y-1,x-1),l3(rdzw,kk,y-1,x))
        correction = mul(mul(mul(derivative,r(.25)),sy),invdz)
        return sub(value,mul(mm,sub(mul(rdy,sub(hat("u",kk),hat("u",kk,True))),correction)))

    def wavg(k):
        kk = jnp.minimum(k,nz-1)
        value = sum4(l3(w,kk),l3(w,kk,y,x-1),l3(w,kk,y-1,x),l3(w,kk,y-1,x-1))
        value = add(value,l3(w,kk+1))
        value = add(value,l3(w,kk+1,y,x-1))
        value = add(value,l3(w,kk+1,y-1,x))
        return mul(r(.125),add(value,l3(w,kk+1,y-1,x-1)))

    def height(k):
        def z(yy,xx):
            return sub(div(add(l3(ph,k,yy,xx),l3(phb,k,yy,xx)),r(9.81)),l2(ht,yy,xx))
        return mul(r(.25),sum4(z(y,x),z(y,x-1),z(y-1,x),z(y-1,x-1)))

    def integrate(k,s):
        value,use,vl,wl,zl = s
        vu,wu,zu = vort(k+1),wavg(k+1),height(k+1)
        band = (zl >= r(2000)) & (zu <= r(5000))
        positive = (wl > r(0)) & (wu > r(0))
        term = mul(mul(add(mul(wl,vl),mul(wu,vu)),r(.5)),sub(zu,zl))
        value = jnp.where(band,jnp.where(positive,add(value,term),r(0)),value)
        return value,use & ~(band & ~positive),vu,wu,zu

    zeros = jnp.zeros((BLOCK,),r)
    value,use,_,_,_ = lax.fori_loop(0,nz,integrate,(zeros,jnp.ones((BLOCK,),jnp.bool_),vort(jnp.int32(0)),wavg(jnp.int32(0)),height(jnp.int32(0))))
    retained = pt.load(old_uh.at[y,x],mask=valid,other=r(0))
    pt.store(uh_out.at[y,x],jnp.where(active,value,retained),mask=valid)
    pt.store(use_out.at[y,x],(use & active).astype(jnp.int32),mask=valid)


def _helicity_smooth(old,uh,use,out,*,ny,nx,interpret):
    lane = pl.program_id(0)*BLOCK+jnp.arange(BLOCK,dtype=jnp.int32)
    y,x = lane//nx,lane%nx
    valid = lane < ny*nx
    # WRF's west then south boundary copy; eastern/northern mass edges are
    # unsmoothed because the source's ide/jde denote staggered outer edges.
    yy,xx = jnp.maximum(y,1),jnp.maximum(x,1)
    interior = valid & (yy < ny-1) & (xx < nx-1)
    add,_,mul,_ = _real_ops(interpret)
    r = jnp.float32
    def h(dy,dx):
        return pt.load(uh.at[yy+dy,xx+dx],mask=interior,other=r(0))
    value = mul(r(.25),h(0,0))
    cross = add(add(add(h(0,1),h(0,-1)),h(1,0)),h(-1,0))
    diagonal = add(add(add(h(1,1),h(-1,1)),h(1,-1)),h(-1,-1))
    value = add(add(value,mul(r(.125),cross)),mul(r(.0625),diagonal))
    previous = pt.load(old.at[yy,xx],mask=valid,other=r(0))
    selected = pt.load(use.at[yy,xx],mask=interior,other=jnp.int32(0)) > jnp.int32(0)
    value = jnp.where(interior & selected & (value > previous),value,previous)
    pt.store(out.at[y,x],value,mask=valid)


def update_helicity(diagnostics,*,u,v,w,ph,phb,msfux,msfvy,terrain,rdx,rdy,
                    dn,dnw,fnm,fnp,cf1,cf2,cf3,zx=None,zy=None,rdzw=None,interpret=None):
    """Literal cal_helicity for a full specified/nested nonperiodic domain.

    Run before RK, then call ``update_nwp`` after RK. This entry point needs
    at least three mass levels; the 2–5 km integration and the nine-point
    smoother follow pristine operation order, including the source's
    all-positive-column test and asymmetric physical-edge ownership.
    """
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    nz,ny,nx = w.shape[0]-1,w.shape[1],w.shape[2]
    if nz < 3 or ny < 3 or nx < 3:
        raise ValueError("WRF helicity needs three mass levels and a 3x3 horizontal grid")
    if zx is None or zy is None or rdzw is None:
        zx,zy,rdzw = helicity_metrics(ph,phb,rdx=rdx,rdy=rdy)
    constants = jnp.stack(tuple(jnp.asarray(a,jnp.float32) for a in (rdx,rdy,cf1,cf2,cf3)))
    arrays = tuple(jnp.asarray(a,jnp.float32) for a in (u,v,w,ph,phb,msfux,msfvy,terrain,rdzw,zx,zy,fnm,fnp,dn,dnw))
    real = jax.ShapeDtypeStruct((ny,nx),jnp.float32)
    integer = jax.ShapeDtypeStruct((ny,nx),jnp.int32)
    grid = ((ny*nx+BLOCK-1)//BLOCK,)
    uh,use = pl.pallas_call(partial(_helicity_kernel,nz=nz,ny=ny,nx=nx,interpret=interpret),
                out_shape=(real,integer),grid=grid,interpret=interpret,name="nwp_helicity_column",
                compiler_params=pt.CompilerParams(num_warps=4))(diagnostics.uh,*arrays,constants)
    maximum = pl.pallas_call(partial(_helicity_smooth,ny=ny,nx=nx,interpret=interpret),
                out_shape=real,grid=grid,interpret=interpret,name="nwp_helicity_smooth",
                compiler_params=pt.CompilerParams(num_warps=4))(diagnostics.up_heli_max,uh,use)
    return diagnostics._replace(up_heli_max=maximum,uh=uh)

