"""Operational WRF lake tiles: compact lake columns, explicit Noah-MP bypass."""
from dataclasses import dataclass
from typing import NamedTuple
import jax
from jax import numpy as jnp
import numpy as np

from gpuwrf.physics.lake_initialization import LakeColumnState
from gpuwrf.physics.lake_step import LakeAtmosphere,LakeSurface,lake_step


class LakeCarry(NamedTuple):
    columns: LakeColumnState
    surface: LakeSurface
    energy_residual: object
    background: object  # packed full-grid GRDFLX/ALBEDO; held, never recomputed at output


@dataclass(frozen=True,eq=False)
class LakeBundle:
    shape: tuple[int,int]
    indices: tuple[int,...]
    nonlake_indices: tuple[int,...]
    initial: LakeCarry
    baseline: object
    latitude: object
    emissivity: object
    lakemask: object
    xland: object


def select_columns(value,bundle,indices=None):
    indices = bundle.indices if indices is None else indices
    value = jnp.asarray(value)
    return value.reshape((np.prod(bundle.shape),)+value.shape[2:])[jnp.asarray(indices,jnp.int32)]


def scatter_columns(value,columns,bundle):
    value = jnp.asarray(value)
    flat = value.reshape((np.prod(bundle.shape),)+value.shape[2:])
    return flat.at[jnp.asarray(bundle.indices,jnp.int32)].set(jnp.asarray(columns,value.dtype)).reshape(value.shape)


def noahmp_without_lakes(kernel,land,forcing,static,dt,bundle,*,energy_params,rad_params,history=False,frozen_land=None):
    """Execute the land model only on non-lake points, then restore full grid.

    Partition indices are frozen during ingestion. No dynamic/nonzero selection
    or host roundtrip occurs in the timestep. Lake points never enter kernel.
    """
    from gpuwrf.contracts.noahmp_state import NoahMPFluxes
    if not bundle.indices:
        return kernel(land,forcing,static,dt,energy_params=energy_params,rad_params=rad_params,**({'history':True} if history else {}))
    frozen_land = land if frozen_land is None else frozen_land
    shape = bundle.shape
    count = len(bundle.nonlake_indices)
    zero = jnp.zeros(shape,dtype=land.t_skin.dtype)
    if count == 0:
        from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS
        nm = NoahMPFluxes(*(zero for _ in range(14)),history=({name:zero for name in LAND_FLUX_FIELDS} if history else None))
        return frozen_land,nm
    def pack(x):
        # Noah-MP payload/profile/band arrays are layer-major, unlike the lake
        # component's last-axis layers. Preserve that convention explicitly.
        if hasattr(x,'shape') and x.ndim >= 2 and tuple(x.shape[-2:]) == shape:
            selected=x.reshape(x.shape[:-2]+(np.prod(shape),))[...,jnp.asarray(bundle.nonlake_indices,jnp.int32)]
            # Retain both horizontal axes: pristine Noah-MP ports use
            # dz_soil[:,None,None] and thus require (level,ny,nx), even on tiles.
            return selected.reshape(x.shape[:-2]+(1,count))
        return x
    args = [jax.tree.map(pack,x) for x in (land,forcing,static,energy_params,rad_params)]
    new,flux = kernel(args[0],args[1],args[2],dt,energy_params=args[3],rad_params=args[4],**({'history':True} if history else {}))
    nonlake = jnp.asarray(bundle.nonlake_indices,jnp.int32)
    def restore(original,part):
        if hasattr(original,'shape') and original.ndim >= 2 and tuple(original.shape[-2:]) == shape:
            flat = original.reshape(original.shape[:-2]+(np.prod(shape),))
            return flat.at[...,nonlake].set(part.reshape(original.shape[:-2]+(count,))).reshape(original.shape)
        return part
    new = jax.tree.map(restore,frozen_land,new)
    def expand(x):
        if hasattr(x,'shape') and x.ndim >= 2 and x.shape[-2:] == (1,count):
            flat = jnp.zeros(x.shape[:-2]+(np.prod(shape),),x.dtype)
            return flat.at[...,nonlake].set(x.reshape(x.shape[:-2]+(count,))).reshape(x.shape[:-2]+shape)
        return x
    return new,jax.tree.map(expand,flux)


def lake_radiation_land(land,carry,bundle):
    """WRF radiation consumes the held lake ALBEDO/TSK instead of LSM values."""
    if land is None:
        return land
    return land.replace(albedo=scatter_columns(land.albedo,carry.surface.albedo,bundle),
                         t_skin=scatter_columns(land.t_skin,carry.surface.tsk,bundle))


def lake_surface_step(state,carry,bundle,*,radiation,precipitation,grid,dt,history=None):
    if not bundle.indices:
        if history is not None:
            # Lake-free nests still produce current Noah-MP surface history.
            background = jnp.stack((history['grdflx'],history['albedo'])).astype(jnp.float32)
            carry = carry._replace(background=background)
        return state,carry,history
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.coupling.physics_couplers import _wrf_phy_prep_p8w,_wrf_phy_prep_rho_from_state
    from gpuwrf.physics.surface_layer import mynn_driver_surface_fluxes,_mynn_surface_exner_kwargs
    view = _build_column_view(state,grid)
    f32 = jnp.float32
    z_at_w = jnp.asarray(state.ph,f32)/f32(9.81)
    p8w = _wrf_phy_prep_p8w(jnp.asarray(state.p,f32),z_at_w,grid.metrics.fnm,grid.metrics.fnp)
    take = lambda a:select_columns(a,bundle)
    rainbl = (jnp.asarray(precipitation.prcpconv,f32)+jnp.asarray(precipitation.prcpnonc,f32))*f32(dt)
    atmosphere = LakeAtmosphere(take(view.t_air[...,0]),take(p8w[1]),take(p8w[0]),take(view.dz[...,0]),
                                take(view.qv[...,0]),take(view.u[...,0]),take(view.v[...,0]),
                                take(radiation.lwdn),take(bundle.emissivity),take(rainbl),
                                jnp.full((len(bundle.indices),),dt,f32),take(radiation.soldn),carry.surface.albedo,
                                jnp.asarray(bundle.latitude,f32))
    result = lake_step(carry.columns,atmosphere)
    sf = result.surface
    put = lambda old,new:scatter_columns(old,new,bundle)
    hfx,qfx,tsk = put(state.hfx,sf.hfx),put(state.qfx,sf.qfx),put(state.t_skin,sf.tsk)
    rho3d = _wrf_phy_prep_rho_from_state(state,grid.metrics)
    density = jnp.asarray(rho3d[0],f32)
    flux = mynn_driver_surface_fluxes(hfx,qfx,density,jnp.asarray(state.qv[0],f32),tsk,jnp.asarray(view.p[...,0],f32),
                                     **_mynn_surface_exner_kwargs(view,f32))
    # Lake does not export its local friction velocity/roughness/momentum.
    updates = dict(hfx=hfx,qfx=qfx,t_skin=tsk,
                   theta_flux=put(state.theta_flux,take(flux.flt)),
                   qv_flux=put(state.qv_flux,take(flux.flqv)),fltv=put(state.fltv,take(flux.fltv)))
    state = state.replace(**updates)
    if history is not None:
        history = dict(history)
        for name in ('hfx','lh','t2','q2'):
            history[name] = put(history[name],getattr(sf,name))
        history['grdflx'] = put(history['grdflx'],sf.grdflx)
        history['albedo'] = put(history['albedo'],sf.albedo)
        background = jnp.stack((history['grdflx'],history['albedo'])).astype(f32)
    else:
        background = carry.background
    return state,LakeCarry(result.columns,sf,result.energy_residual.astype(f32),background),history


LAKE_HISTORY_FIELDS = tuple(name.upper() for name in LakeColumnState._fields)


def lake_history_fields(carry,bundle):
    """Output-boundary expansion only; WRF has surface then vertical axes."""
    result = {}
    for name,compact in carry.columns._asdict().items():
        full = scatter_columns(jnp.asarray(getattr(bundle.baseline,name)),compact,bundle)
        result[name.upper()] = jnp.moveaxis(full,-1,0) if full.ndim == 3 else full
    # Held lake surface fields overwrite the water/LSM history only on lakes.
    for name in LakeSurface._fields:
        result['_LAKE_'+name.upper()] = getattr(carry.surface,name)
    return result


def overlay_lake_history(diagnostics,carry,bundle):
    diagnostics = dict(diagnostics or {})
    diagnostics.update(GRDFLX=carry.background[0],ALBEDO=carry.background[1])
    fields = lake_history_fields(carry,bundle)
    for name in LakeSurface._fields:
        key = name.upper()
        compact = fields.pop('_LAKE_'+key)
        if key in diagnostics:
            diagnostics[key] = scatter_columns(diagnostics[key],compact,bundle)
        else:
            # New surface outputs GRDFLX/TH2/ALBEDO need existing full-grid
            # values supplied by the normal history producer.
            diagnostics['_LAKE_'+key] = compact
    diagnostics.update(fields)
    diagnostics['_LAKE_BUNDLE'] = bundle
    return diagnostics
