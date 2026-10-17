"""PSU surface analysis nudging, pristine module_fdda_psufddagd.F:693–1320.

Surface tendencies precede the three dimensional terms in FDDAGD. Winds and
theta use the FIRST model level innovation throughout the PBL; moisture uses
each level's innovation and the original WRF saturation restriction.
"""
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.fdda_grid import (
    GridFddaOptions, GridFddaInputs, GridFddaCarry, GridFddaHistoryCarry, _wrf_div, wrf_minutes,
    _kpbl_weights, _face_avg_x, _face_avg_y, _agl_x, _agl_y, _wzfac,
)


@dataclass(frozen=True)
class SurfaceFddaOptions(GridFddaOptions):
    surface_code: int = 1
    interval_m_sfc: int = 60
    guv_sfc: float = 0.0
    gt_sfc: float = 0.0
    gq_sfc: float = 0.0
    rinblw: float = 0.0
    dx_m: float = 3000.0

    def __post_init__(self):
        super().__post_init__()
        if self.surface_code not in (1, 2):
            raise ValueError('grid_sfdda must be 1 or 2')
        if self.interval_m_sfc <= 0:
            raise ValueError('sgfdda_interval_m must be > 0')
        if not np.isfinite(self.rinblw) or self.rinblw < .001:
            raise ValueError('rinblw must be >= 0.001 km')
        if not np.isfinite(self.dx_m) or self.dx_m <= 0:
            raise ValueError('surface FDDA requires positive dx')
        for name in ('guv_sfc', 'gt_sfc', 'gq_sfc'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be positive')


SURFACE_FIELDS = ('u10', 'v10', 'th2', 'q2', 'odis')


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SurfaceAnalysis:
    """Auxinput9 pair, spatially mass-shaped except the U10/V10 staggering."""
    u10_old: Any
    u10_new: Any
    v10_old: Any
    v10_new: Any
    th2_old: Any
    th2_new: Any
    q2_old: Any
    q2_new: Any
    odis_old: Any
    odis_new: Any

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SurfaceGridFddaInputs(GridFddaInputs):
    surface: SurfaceAnalysis | None = None
    surface_steps: tuple[int, ...] = ()

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        return children + (self.surface,), aux + (self.surface_steps,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        base = GridFddaInputs.tree_unflatten(aux[:2], children[:-1])
        return cls(**vars(base), surface=children[-1], surface_steps=aux[2])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SurfaceGridFddaCarry(GridFddaHistoryCarry):
    surface: SurfaceAnalysis | None = None
    sda_hfx: Any = None
    sda_qfx: Any = None
    surface_cursor: Any = None


def seed_surface_carry(inputs, base, dtype):
    first = jax.tree.map(lambda a: jnp.asarray(a[0], dtype), inputs.surface)
    shape = first.th2_old.shape
    members = dict(vars(base))
    members.setdefault("hfx_fdda", jnp.zeros_like(base.t_old))
    return SurfaceGridFddaCarry(**members, surface=first,
        sda_hfx=jnp.zeros(shape, dtype), sda_qfx=jnp.zeros(shape, dtype),
        surface_cursor=jnp.asarray(0, jnp.int32))


def validate_surface_restart(group,namelist,*,required=False):
    """Host E78 admission: never silently recreate a missing auxinput9 cursor."""
    from collections.abc import Mapping
    inputs=getattr(namelist,'data_assimilation',None)
    active=required or isinstance(inputs,SurfaceGridFddaInputs)
    if isinstance(namelist,Mapping):
        raw=(namelist.get('fdda') or {}).get('grid_sfdda',0)
        active=active or any(int(v) for v in (raw if isinstance(raw,(list,tuple)) else (raw,)))
    if not active and not isinstance(group,SurfaceGridFddaCarry):return
    if not isinstance(group,SurfaceGridFddaCarry) or not isinstance(group.surface,SurfaceAnalysis):
        raise ValueError('restart surface FDDA is missing auxinput9 analysis/cursor (E78)')
    for f in fields(group):
        if getattr(group,f.name,None) is None:
            raise ValueError(f'restart surface FDDA lacks {f.name} (E78)')
    index=np.asarray(group.surface_cursor)
    if index.shape!=() or index.dtype!=np.int32 or int(index)<0:
        raise ValueError('restart surface FDDA cursor shape/dtype invalid (E78)')
    shape=np.shape(group.t_old)
    if len(shape)!=3:raise ValueError('restart surface FDDA domain shape invalid (E78)')
    nz,ny,nx=shape
    dtype=np.asarray(group.t_old).dtype
    if dtype not in (np.float32,np.float64):raise ValueError('restart surface FDDA REAL dtype invalid (E78)')
    for f in fields(GridFddaHistoryCarry):
        a=np.asarray(getattr(group,f.name))
        expected=(nz,ny,nx+1) if f.name.startswith(('u_','rundg')) else (nz,ny+1,nx) if f.name.startswith(('v_','rvndg')) else shape
        if a.shape!=expected or a.dtype!=dtype or not np.isfinite(a).all():
            raise ValueError(f'restart surface FDDA {f.name} shape/dtype/nonfinite invalid (E78)')
    for f in fields(group.surface):
        a=np.asarray(getattr(group.surface,f.name))
        expected=(ny,nx+1) if f.name.startswith('u10') else (ny+1,nx) if f.name.startswith('v10') else (ny,nx)
        if a.shape!=expected or a.dtype!=dtype or not np.isfinite(a).all():
            raise ValueError(f'restart surface FDDA {f.name} shape/dtype/nonfinite invalid (E78)')
    for n in ('sda_hfx','sda_qfx'):
        a=np.asarray(getattr(group,n))
        if a.shape!=(ny,nx) or a.dtype!=np.asarray(group.t_old).dtype or not np.isfinite(a).all():
            raise ValueError(f'restart surface FDDA {n} shape/dtype/nonfinite invalid (E78)')
    if isinstance(inputs,SurfaceGridFddaInputs):
        if int(index)>=len(inputs.surface_steps):raise ValueError('restart surface FDDA cursor out of range (E78)')
        for f in fields(group.surface):
            ref=np.asarray(getattr(inputs.surface,f.name)[int(index)])
            if not np.array_equal(np.asarray(getattr(group.surface,f.name)),ref):
                raise ValueError(f'restart surface FDDA {f.name} differs from bound input stream (E78)')


def read_surface_record(carry, inputs, itimestep, dtype):
    hit = jnp.asarray(itimestep, jnp.int32) - 1 == jnp.asarray(inputs.surface_steps, jnp.int32)
    index = jnp.argmax(hit).astype(jnp.int32)
    def load(c):
        pair = jax.tree.map(lambda a: jnp.asarray(jax.lax.dynamic_index_in_dim(
            a, index, axis=0, keepdims=False), dtype), inputs.surface)
        return c.replace(surface=pair, surface_cursor=index)
    return jax.lax.cond(jnp.any(hit), load, lambda c: c, carry)


def density_weight(odis, options, dtype):
    """SFDDAGD observation-density weight + one SMTHER pass, physical edges held."""
    r = lambda x: jnp.asarray(x, dtype)
    rindx = _wrf_div(r(options.rinblw) * r(1000), r(options.dx_m), dtype)
    m = _wrf_div(r(-.8) * r(2), rindx, dtype)
    x = jnp.minimum(_wrf_div(r(odis) * r(1000), r(options.dx_m), dtype), rindx)
    a = jnp.where(r(odis) < r(.5) * r(options.rinblw), r(1), m*x + r(1.8))
    avg = (((a[1:-1, 2:] + a[1:-1, :-2]) + a[2:, 1:-1]) + a[:-2, 1:-1]) * r(.25)
    a = a.at[1:-1, 1:-1].set(a[1:-1, 1:-1] + r(.5)*(avg-a[1:-1, 1:-1]))
    return jnp.clip(a, r(0), r(1))  # literal WRF WHERE restrictions


def surface_time_factors(o, lead_seconds, dtype):
    r = lambda x: jnp.asarray(x, dtype)
    xtime, mins = wrf_minutes(lead_seconds, dtype)
    interval = r(o.interval_m_sfc)
    old = jnp.floor(_wrf_div(mins, interval, dtype)) * interval * r(1)
    coef = _wrf_div(mins-old, (old+interval)-old, dtype)
    end = r(o.end_h_sfc)*r(60) + r(abs(o.dtramp_min))  # unconditional caller assignment :235
    tfac = r(1)
    if o.if_ramping == 1 and abs(o.dtramp_min) > 0:
        end = r(o.end_h_sfc)*r(60) + r(max(o.dtramp_min, 0))
        tfac = jnp.where(xtime < end-r(abs(o.dtramp_min)), r(1),
            jnp.where(xtime <= end, _wrf_div(end-xtime, r(abs(o.dtramp_min)), dtype), r(0)))
    if o.if_ramping == 1 and o.dtramp_min > 0:
        coef = jnp.where((xtime >= end-r(abs(o.dtramp_min))) & (xtime <= end), r(0), coef)
    return coef, tfac


def sfddagd(o, pair, u, v, th, qv, t_phy, p_phy, pblh, ht, zw, regime, znt,
            lead_seconds, dtype):
    r = lambda x: jnp.asarray(x, dtype)
    u,v,th,qv,t_phy,p_phy,pblh,ht,zw,regime,znt = map(r,
        (u,v,th,qv,t_phy,p_phy,pblh,ht,zw,regime,znt))
    nz = th.shape[0]
    coef, tfac = surface_time_factors(o, lead_seconds, dtype)
    def interp(a,b):
        return r(a)*(r(1)-coef) + r(b)*coef
    wb = [ _kpbl_weights(_face_avg_x(pblh), _agl_x(zw,ht), dtype),
           _kpbl_weights(_face_avg_y(pblh), _agl_y(zw,ht), dtype),
           _kpbl_weights(pblh, zw-ht[None], dtype)]
    wz = [_wzfac(nz, getattr(o,'if_zfac_'+s), getattr(o,'k_zfac_'+s), dtype)[:,None,None]
          for s in ('uv','t','q')]
    old,new = (density_weight(a,o,dtype) for a in (pair.odis_old,pair.odis_new))
    blw = [interp(_face_avg_x(old),_face_avg_x(new)),
           interp(_face_avg_y(old),_face_avg_y(new)), interp(old,new)]
    z = r(.5)*(zw[:-1]+zw[1:])
    def windcor(reg,difz,z0):
        # Benign operands only outside WRF's IF. An active negative roughness
        # retains WRF's invalid arithmetic; never repair an active operand.
        active=(reg > r(3.5)) & (difz > r(10))
        safez = jnp.where(active,difz,r(40))
        safe0 = jnp.where(active,z0,r(1))
        corr = jnp.where(z0 <= r(.2),r(1)+r(.320)*safe0**r(.2),r(1.169)+r(.315)*z0)
        fac = r(.5)*(_wrf_div(jnp.log(_wrf_div(safez,r(.05),dtype)),jnp.log(r(40)/r(.05)),dtype)
                    + _wrf_div(jnp.log(safez),jnp.log(r(40)),dtype))
        return jnp.where((reg > r(3.5)) & (difz > r(10)),fac*corr,r(1))
    cu = windcor(_face_avg_x(regime),_agl_x(z,ht)[0],_face_avg_x(znt))
    cv = windcor(_face_avg_y(regime),_agl_y(z,ht)[0],_face_avg_y(znt))
    au = interp(pair.u10_old,pair.u10_new)[...,1:-1]*cu
    av = interp(pair.v10_old,pair.v10_new)[...,1:-1,:]*cv
    ru = r(o.guv_sfc)*(r(1)-wb[0])*wz[0]*tfac*blw[0]*(au-u[0,...,1:-1])
    rv = r(o.guv_sfc)*(r(1)-wb[1])*wz[0]*tfac*blw[1]*(av-v[0,...,1:-1,:])
    es = r(.6112)*jnp.exp(_wrf_div(r(17.67)*(t_phy-r(273.15)),t_phy-r(29.65),dtype))*r(10)
    ep2 = r(np.float32(np.float32(287)/np.float32(461.6)) if dtype==jnp.float32 else 287/461.6)
    qsat = _wrf_div(ep2*es,_wrf_div(p_phy,r(100),dtype)-es,dtype)
    qa = jnp.minimum(interp(pair.q2_old,pair.q2_new), qsat)
    wt = wq = r(1)-wb[2]
    stab = r(1)
    if o.surface_code == 2:
        wt = wq = (jnp.arange(nz) == 0)[:,None,None].astype(dtype)
        stab = jnp.where(regime <= r(1.1),_wrf_div(r(1),r(3),dtype),r(1))
    rt = stab*r(o.gt_sfc)*wt*wz[1]*tfac*blw[2]*(interp(pair.th2_old,pair.th2_new)-th[0])
    rq = stab*r(o.gq_sfc)*wq*wz[2]*tfac*blw[2]*(qa-qv)
    sda_h = _wrf_div(rt[0],(_wrf_div(r(1e5),p_phy[0],dtype))**_wrf_div(r(287),r(1004),dtype),dtype)
    return (jnp.zeros_like(u).at[...,1:-1].set(ru),
            jnp.zeros_like(v).at[...,1:-1,:].set(rv),rt,rq), (sda_h,rq[0])
