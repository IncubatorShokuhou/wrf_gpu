"""Literal Noah-MP BEP/BEM rural blend and multi-layer PBL source interface."""
from typing import NamedTuple

import jax.numpy as jnp


class UrbanPblSources(NamedTuple):
    a_u: object
    a_v: object
    a_t: object
    a_q: object
    a_e: object
    b_u: object
    b_v: object
    b_t: object
    b_q: object
    b_e: object
    ground_length: object
    building_length: object
    surface: object
    volume: object
    fraction: object


class RuralSurface(NamedTuple):
    skin: object
    heat: object
    moisture: object
    ground: object
    albedo: object
    emissivity: object
    ust: object


class UrbanSurface(NamedTuple):
    skin: object
    heat: object
    moisture: object
    latent: object
    ground: object
    albedo: object
    emissivity: object
    ust: object
    urban_skin: object
    urban_heat: object
    urban_latent: object
    urban_ground: object
    urban_net_radiation: object


def noahmp_bep_blend(result,forcing,rural:RuralSurface,fraction,*,raw_moisture=None):
    """module_sf_noahmpdrv.F:3703–3774, including WRF's CP vs BEP CP_U.

    Pure BEP raw moisture is zero; BEP+BEM supplies its layer moisture source.
    Sources are returned before any PBL-specific length weighting. A_E/A_Q are
    explicitly zeroed by WRF here, even if a canopy kernel had supplied them.
    """
    source=result.sources;dtype=forcing.theta.dtype
    cp=jnp.asarray(7.*287./2.,dtype);xlv=jnp.asarray(2.5e6,dtype);sigma=jnp.asarray(5.67e-8,dtype)
    frc=fraction[...,None];dz=forcing.z[...,1:]-forcing.z[...,:-1]
    au=source.a_u*frc;av=source.a_v*frc;at=source.a_t*frc
    bu=source.b_u*frc;bv=source.b_v*frc;bt=source.b_t*frc;be=source.b_e*frc
    bq=jnp.zeros_like(bt) if raw_moisture is None else raw_moisture*frc
    hfx=jnp.zeros_like(fraction);qfx=jnp.zeros_like(fraction);umom=jnp.zeros_like(fraction);vmom=jnp.zeros_like(fraction)
    for k in range(dz.shape[-1]):
        hfx=hfx+bt[...,k]*forcing.rho[...,k]*cp*dz[...,k]*source.volume[...,k]
        qfx=qfx+bq[...,k]*dz[...,k]*source.volume[...,k]
        umom=umom+(au[...,k]*forcing.u[...,k]+bu[...,k])*dz[...,k]*source.volume[...,k]
        vmom=vmom+(av[...,k]*forcing.v[...,k]+bv[...,k])*dz[...,k]*source.volume[...,k]
    volume=(1.-frc)+source.volume*frc
    surface=(1.-frc)+source.surface[...,:-1]*frc
    speed=(forcing.u[...,0]**2+forcing.v[...,0]**2.)**.5
    drag=(1.-fraction)*(-rural.ust*rural.ust)/dz[...,0]/speed
    au=au.at[...,0].add(drag);av=av.at[...,0].add(drag)
    bt=bt.at[...,0].add((1.-fraction)*rural.heat/dz[...,0]/forcing.rho[...,0]/cp)
    bq=bq.at[...,0].add((1.-fraction)*rural.moisture/dz[...,0]/forcing.rho[...,0])
    umom=(1.-fraction)*rural.ust*rural.ust*forcing.u[...,0]/speed+umom
    vmom=(1.-fraction)*rural.ust*rural.ust*forcing.v[...,0]/speed+vmom
    surface=surface.at[...,0].set(1.)
    surface=jnp.concatenate((surface,source.surface[...,-1:]),axis=-1)
    rural_lw=-rural.emissivity*sigma*rural.skin**4.-(1.-rural.emissivity)*forcing.lw
    total_lw=(1.-fraction)*rural_lw+fraction*result.upward_longwave
    emissivity=(1.-fraction)*rural.emissivity+fraction*result.emissivity
    urban_skin=jnp.maximum(0.,(-result.upward_longwave-(1.-result.emissivity)*forcing.lw)/result.emissivity/sigma)**.25
    skin=jnp.maximum(0.,(-1.*total_lw-(1.-emissivity)*forcing.lw)/emissivity/sigma)**.25
    shortwave=(1.-fraction)*forcing.sw*(1.-rural.albedo)+fraction*result.absorbed_shortwave
    albedo=jnp.where(forcing.sw>0.,1.-shortwave/jnp.where(forcing.sw>0.,forcing.sw,1.),rural.albedo)
    ground=(1.-fraction)*rural.ground+fraction*result.ground_flux*(-1.)
    moisture=(1.-fraction)*rural.moisture+qfx
    heat=hfx+(1-fraction)*rural.heat
    safe_frc=jnp.where(fraction>0.,fraction,1.)
    output=UrbanSurface(skin,heat,moisture,moisture*xlv,ground,albedo,emissivity,
        (umom**2.+vmom**2.)**.25,urban_skin,hfx/safe_frc,qfx*xlv/safe_frc,result.ground_flux,
        result.absorbed_shortwave+result.emissivity*forcing.lw-result.upward_longwave)
    pbl=UrbanPblSources(au,av,at,jnp.zeros_like(at),jnp.zeros_like(at),bu,bv,bt,bq,be,
                       result.ground_length,result.building_length,surface,volume,fraction)
    return output,pbl
