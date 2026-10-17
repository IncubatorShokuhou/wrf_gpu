"""Pristine-WRF BEP/BEM component kernels used by the full canopy models.

REAL arithmetic follows module_sf_bep.F and module_sf_bep_bem.F; HVAC follows
module_sf_bem.F. Arrays use trailing level axes (wall fluxes: side, level).
Full tables/radiation/building steps live in urban_bep_model/urban_bem_model;
the operational Noah-MP/PBL caller lives in coupling.urban_multilayer_hook.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class FlatFlux(NamedTuple):
    u: object
    v: object
    heat: object
    tke: object


class WallFlux(NamedTuple):
    a_u: object
    a_v: object
    b_u: object
    b_v: object
    a_t: object
    b_t: object
    b_e: object


def _louis(dz, z0, u, v, theta, surface):
    speed = jnp.maximum((u**2 + v**2)**0.5, 0.01)
    zz = dz / 2.
    ri = 2.*9.81*zz*(theta-surface)/((theta+surface)*(speed**2))
    aa = 0.40/jnp.log(zz/z0)
    stable = 1/(1+0.5*9.4*ri)**2
    c = 9.4*7.4*aa*aa*(zz/z0)**0.5
    # Inactive branch evaluated by where: protect sqrt without changing ri.
    unstable = jnp.maximum(-ri, 0.)**0.5
    fm = jnp.where(ri > 0., stable, 1-9.4*ri/(1+c*unstable))
    c = c*5.3/7.4
    fh = jnp.where(ri > 0., stable, 1-9.4*ri/(1+c*unstable))
    fbuw = -aa*aa*speed*speed*fm
    fbpt = -aa*aa*speed*(theta-surface)*fh/0.74
    ustar = (-fbuw)**0.5
    tstar = -fbpt/ustar
    return speed, aa, fh, ustar, tstar


def bep_flux_flat(dz, z0, u, v, theta, theta0, surface):
    """BEP :2544-2664; fluxes in m²/s², K m/s, and m²/s³."""
    speed, _, _, us, ts = _louis(dz, z0, u, v, theta, surface)
    return FlatFlux(-us*us*u/speed, -us*us*v/speed, -us*ts,
                    -9.81/theta0*us*ts)


def _wall_momentum(u, v, angle, drag):
    ortho = jnp.abs(jnp.cos(angle)*u-jnp.sin(angle)*v)
    return (-drag*ortho/2.*jnp.cos(angle)*jnp.cos(angle),
            -drag*ortho/2.*jnp.sin(angle)*jnp.sin(angle),
            drag*ortho/2.*jnp.sin(angle)*jnp.cos(angle)*v,
            drag*ortho/2.*jnp.sin(angle)*jnp.cos(angle)*u,
            drag*jnp.abs(ortho)**3./2.)


def bep_flux_wall(u, v, theta, rho, surface, angle, dt):
    """BEP :2466-2540; explicit heat term retains WRF's two divisions."""
    au, av, bu, bv, be = _wall_momentum(u, v, angle, 0.4)
    speed = (u**2+v**2)**0.5
    hc = jnp.minimum(5.678*(1.09+0.23*(speed/0.3048)), rho*1004./dt)
    if jnp.asarray(theta).dtype == jnp.float32:
        cp = jax.lax.optimization_barrier(jnp.asarray(1004.,jnp.float32))
        first = hc*surface/rho/cp
        second = hc/rho/cp*theta
        bt = jnp.nextafter(first,first)-jnp.nextafter(second,second)
    else:
        bt = hc*surface/rho/1004.-hc/rho/1004.*theta
    return WallFlux(au, av, bu, bv, jnp.zeros_like(bt), bt, be)


def bem_flux_flat(dz, z0, u, v, theta, theta0, surface, qv, rho, pressure):
    """BEP+BEM :4596-4716; sensible and latent heat in W/m² (WRF signs)."""
    speed, aa, fh, us, ts = _louis(dz, z0, u, v, theta, surface)
    rcp = jnp.asarray(287./1004., dtype=jnp.asarray(theta).dtype)
    # Preserve WRF REAL division; folding /1e5 to *(rounded 1e-5) changes
    # saturation enough to fail the oracle (E123). Product barriers do not fix it.
    p0 = jax.lax.optimization_barrier(jnp.asarray(1.e5, dtype=jnp.asarray(theta).dtype))
    tmp = surface*(pressure/p0)**rcp-273.15
    es = 6.11*(10**(tmp*7.5/(237.7+tmp)))
    saturated = 0.62197*es/(0.01*pressure-0.378*es)
    resistance = 0.74/(aa*aa*speed*fh)
    fbqq = -(qv-saturated)/resistance
    qs = -fbqq/us
    return (-us*us*u/speed, -us*us*v/speed, us*ts*rho*1004.,
            us*qs*rho*2.45e6, -9.81/theta0*us*ts)


def bem_flux_wall(u, v, theta, rho, surface, window, angle, dt, drag):
    """BEP+BEM :4108-4199; wall/window heat flux in W/m², wind break at 4.88."""
    au, av, bu, bv, be = _wall_momentum(u, v, angle, drag)
    speed = (u**2+v**2)**0.5
    hc = jnp.where(speed < 4.88, 5.678*(1.09+0.23*(speed/0.3048)),
                   5.678*0.53*((speed/0.3048)**0.78))
    hw = jnp.where(speed < 4.88, 5.678*(0.99+0.21*(speed/0.3048)),
                   5.678*0.50*((speed/0.3048)**0.78))
    hc = jnp.minimum(hc, rho*1004./dt)
    hw = jnp.minimum(hw, rho*1004./dt)
    return au, av, bu, bv, hc*(theta-surface), hw*(theta-window), be


def bep_soil_temp(depth, temp, diffusivity, capacity, sw, lw, pressure, dt,
                  emissivity, albedo, sensible):
    """BEP :2318-2460; interior-first material layers, surface at last index.

    Returns (temperature[K], surface potential temperature[K], radiation[W/m²],
    heat flux[W/m²]). WRF eliminates from the surface toward the interior first.
    The small static material dimension is unrolled; no domain-wide scan/carry.
    """
    n = temp.shape[-1]
    rn = (lambda x: jnp.nextafter(x,x)) if temp.dtype == jnp.float32 else (lambda x:x)
    if temp.dtype == jnp.float32:
        # WRF evaluates these REAL coefficients in the routine. XLA's HLO
        # constant evaluator otherwise changes thousands of material results.
        depth, diffusivity, capacity, dt = jax.lax.optimization_barrier(
            (depth, diffusivity, capacity, jnp.asarray(dt,temp.dtype)))
    if n < 2:
        raise ValueError('WRF BEP material column needs at least two layers')
    tsig = temp[..., -1]
    alpha = (1.-albedo)*sw+emissivity*lw-emissivity*5.67e-8*(tsig**4)+sensible
    cddz = [diffusivity[..., 0]/depth[..., 0]]
    for k in range(1, n):
        cddz.append(2.*diffusivity[..., k]/(depth[..., k]+depth[..., k-1]))
    lower = [jnp.zeros_like(tsig)]
    diagonal = [jnp.ones_like(tsig)]
    upper = [jnp.zeros_like(tsig)]
    rhs = [temp[..., 0]]
    for k in range(1, n-1):
        lower.append(-cddz[k]*dt/depth[..., k])
        diagonal.append(1+dt*(cddz[k]+cddz[k+1])/depth[..., k])
        upper.append(-cddz[k+1]*dt/depth[..., k])
        rhs.append(temp[..., k])
    lower.append(-dt*cddz[-1]/depth[..., -1])
    diagonal.append(1.+dt*cddz[-1]/depth[..., -1])
    upper.append(jnp.zeros_like(tsig))
    rhs.append(rn(tsig+rn(dt*alpha/capacity[..., -1]/depth[..., -1])))
    for k in range(n-2, -1, -1):
        rhs[k] = rn(rhs[k]-rn(upper[k]*rhs[k+1]/diagonal[k+1]))
        diagonal[k] = rn(diagonal[k]-rn(upper[k]*lower[k+1]/diagonal[k+1]))
    for k in range(1, n):
        rhs[k] = rn(rhs[k]-rn(lower[k]*rhs[k-1]/diagonal[k-1]))
    result = jnp.stack([rhs[k]/diagonal[k] for k in range(n)], axis=-1)
    rcp = jnp.asarray(287./1004., dtype=temp.dtype)
    if temp.dtype == jnp.float32:
        p0 = jax.lax.optimization_barrier(jnp.asarray(1.e5,temp.dtype))
        pt = result[..., -1]*(pressure/p0)**(-rcp)
    else:
        pt = result[..., -1]*(pressure/1.e5)**(-rcp)
    radiation = (1.-albedo)*sw+emissivity*lw-emissivity*5.67e-8*(tsig**4)
    return result, pt, radiation, alpha


def bep_interpolate(z, urban_z, values):
    """BEP :1080-1127; overlap-weighted layer average, sequential WRF sum."""
    total = jnp.zeros_like(urban_z[..., :-1])
    for k in range(z.shape[-1]-1):
        overlap = jnp.maximum(jnp.minimum(z[..., k+1, None], urban_z[..., 1:])
                              -jnp.maximum(z[..., k, None], urban_z[..., :-1]), 0.)
        total = total+values[..., k, None]*overlap
    denominator = urban_z[..., 1:]-urban_z[..., :-1]
    if total.dtype == jnp.float32:
        denominator = jax.lax.optimization_barrier(denominator)
    return total/denominator


class MesoSources(NamedTuple):
    a_u: object
    a_v: object
    a_t: object
    a_e: object
    b_u: object
    b_v: object
    b_t: object
    b_e: object
    surface: object  # interface array (nmeso+1); WRF's bottom value is zero
    volume: object


def bep_urban_meso(z, dz, urban_z, pb, ss, building_width, street_width,
                   wall: WallFlux, horizontal: FlatFlux, *, nd):
    """BEP :1487-1683; nd is the static number (1 or 2) of street directions.

    pb/ss contain probabilities on urban interfaces; fluxes on urban layers.
    Outputs are per urban area, before the Noah-MP urban-fraction/rural blend.
    """
    if nd not in (1, 2):
        raise ValueError('WRF BEP supports one or two street directions')
    n = urban_z.shape[-1]-1
    zero = jnp.zeros_like(z[..., :-1])
    outputs = [zero for _ in range(8)]
    surface = jnp.zeros_like(z)
    volume = zero
    for d in range(nd):
        bs, ws = building_width[..., d, None], street_width[..., d, None]
        sr = jnp.zeros_like(z[..., 1:])
        for k in range(1, n):
            hit = (z[..., 1:] < urban_z[..., k, None]) & (z[..., 1:] >= urban_z[..., k-1, None])
            sr = jnp.where(hit, pb[..., k, None], sr)
        surface = surface.at[..., 1:].add(((ws+(1.-sr)*bs)/(ws+bs))/nd)
    for d in range(nd):
        bs, ws = building_width[..., d, None], street_width[..., d, None]
        vtot = zero
        for k in range(n):
            overlap = jnp.maximum(jnp.minimum(urban_z[..., k+1, None], z[..., 1:])
                                  -jnp.maximum(urban_z[..., k, None], z[..., :-1]), 0.)
            vtot = vtot+pb[..., k+1, None]*overlap
        vtot = vtot/(z[..., 1:]-z[..., :-1])
        volume = volume+(1.-vtot*bs/(ws+bs))/nd
    for d in range(nd):
        bs, ws = building_width[..., d, None], street_width[..., d, None]
        fact0 = 1./volume[..., 0]/dz[..., 0]*ws[..., 0]/(ws[..., 0]+bs[..., 0])/nd
        for j, flux in enumerate(horizontal):
            # FlatFlux is u,v,heat,tke -> outputs b_u,b_v,b_t,b_e.
            scale = urban_z[..., 1]-urban_z[..., 0] if j == 3 else 1.
            outputs[4+j] = outputs[4+j].at[..., 0].add(flux[..., d, 0]*fact0*scale)
        sums = [zero for _ in range(4)]
        for k in range(1, n):
            hit = (z[..., :-1] <= urban_z[..., k, None]) & (z[..., 1:] > urban_z[..., k, None])
            for j, flux in enumerate(horizontal):
                scale = urban_z[..., k+1]-urban_z[..., k] if j == 3 else 1.
                term = ss[..., k, None]*flux[..., d, k, None]*scale[..., None] if j == 3 else ss[..., k, None]*flux[..., d, k, None]
                sums[j] = sums[j]+jnp.where(hit, term, 0.)
        fact = bs/(ws+bs)/volume/dz/nd
        for j in range(4):
            outputs[4+j] = outputs[4+j]+sums[j]*fact
    # Vertical impacts preserve iz_u accumulation order; all mesoscale levels batch together.
    for d in range(nd):
        bs, ws = building_width[..., d, None], street_width[..., d, None]
        sums = [zero for _ in range(7)]
        for k in range(n):
            overlap = jnp.maximum(jnp.minimum(urban_z[..., k+1, None], z[..., 1:])
                                  -jnp.maximum(urban_z[..., k, None], z[..., :-1]), 0.)
            fact = overlap/(ws+bs)
            for j, flux in enumerate(wall):
                sums[j] = sums[j]+pb[..., k+1, None]*(flux[..., 2*d, k, None]+flux[..., 2*d+1, k, None])*fact
        fact = 1./volume/dz/nd
        # WallFlux order: au,av,bu,bv,at,bt,be. a_e remains WRF's zero.
        for j, target in enumerate((0, 1, 4, 5, 2, 6, 7)):
            outputs[target] = outputs[target]+sums[j]*fact
    return MesoSources(*outputs, surface, volume)


def bep_interp_length(z, urban_z, ss, street_width, building_width, *, nd):
    """BEP :1689-1770; ground and building length scales [m]."""
    n = urban_z.shape[-1]-1
    ulu = jnp.zeros_like(z[..., :-1])
    ssl = jnp.zeros_like(ulu)
    for d in range(nd):
        for k in range(1, n):
            hit = urban_z[..., k, None] > z[..., :-1]
            ulu = ulu+jnp.where(hit, ss[..., k, None]/urban_z[..., k, None]/nd, 0.)
            ssl = ssl+jnp.where(hit, ss[..., k, None]/nd, 0.)
    length = jnp.where(ulu != 0., ssl/jnp.where(ulu != 0., ulu, 1.), 0.)
    ground = jnp.zeros_like(ulu)
    middle = (z[..., :-1]+z[..., 1:])/2.
    for d in range(nd):
        ws, bs = street_width[..., d, None], building_width[..., d, None]
        area = jnp.broadcast_to(ws, middle.shape)
        inverse = ws/middle
        for k in range(n):
            hit = middle > urban_z[..., k, None]
            denominator = jnp.where(hit, middle-urban_z[..., k, None], 1.)
            inverse = inverse+jnp.where(hit, ss[..., k, None]*bs/denominator, 0.)
            area = area+jnp.where(hit, ss[..., k, None]*bs, 0.)
        ground = ground+inverse/area/nd
    return 1./ground, length


def bem_regtemp(switch, hour, dt, heat_capacity, heat, temperature,
                time_on, time_off, target, gap):
    """BEM :2085-2165; required sensible extraction [W], inclusive hours."""
    proposed = temperature+(dt/heat_capacity)*heat
    cooling = proposed > target+gap
    boundary = jnp.where(cooling, target+gap, target-gap)
    need = heat-(heat_capacity/dt)*(boundary-temperature)
    rate = jnp.abs(need-heat)/heat_capacity
    need = jnp.where(rate > 0.001, heat+jnp.where(cooling, 0.001, -0.001)*heat_capacity, need)
    active = (switch != 0) & (hour >= time_on) & (hour <= time_off) & (jnp.abs(proposed-target) > gap)
    return jnp.where(active, need, 0.)


def bem_reghum(switch, hour, dt, volume, rho, latent, heat, humidity,
               time_on, time_off, target, gap):
    """BEM :2170-2253; required latent extraction [W]."""
    mass_heat = latent*rho*volume
    proposed = humidity+(dt/mass_heat)*heat
    drying = proposed > target+gap
    boundary = jnp.where(drying, target+gap, target-gap)
    need = heat-(mass_heat/dt)*(boundary-humidity)
    rate = jnp.abs(need-heat)/mass_heat
    need = jnp.where(rate > 1.e-6, heat+jnp.where(drying, 1.e-6, -1.e-6)*mass_heat, need)
    active = (switch != 0) & (hour >= time_on) & (hour <= time_off) & (jnp.abs(proposed-target) > gap)
    return jnp.where(active, need, 0.)


def bem_air_cond(sensible_need, latent_need, cop):
    """BEM :2258-2312; (sensible exhaust, latent exhaust, power, latent need).

    WRF mutates hlneed to zero during heating or zero sensible demand; callers
    must use the fourth output for the indoor humidity update.
    """
    cooling = sensible_need > 0.
    latent = jnp.where(cooling, latent_need, 0.)
    power = (1./cop)*(jnp.abs(sensible_need)+jnp.abs(latent))
    return (jnp.where(cooling, (1/cop)*(jnp.abs(sensible_need)+jnp.abs(latent))+sensible_need, 0.),
            latent, power, latent)
