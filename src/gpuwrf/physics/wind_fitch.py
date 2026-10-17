"""WRF ``module_wind_fitch.F`` REAL turbine drag, power and QKE source.

Only turbine columns are processed. Winds are the PBL driver's *input* winds;
QKE and DU/DV are its outputs. QKE is twice TKE. POWER is watts per grid cell,
not a time integral. The WRF 1.23 kg/m3 reference density is intentional.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def turbine_area(z1, z2, diameter, density):
    r = diameter / 2.
    a, b = r - z1, r - z2
    c, d = jnp.abs(a), jnp.abs(b)
    c1, c2 = c * jnp.sqrt(r*r - c*c), r*r * jnp.arcsin(c/r)
    d1, d2 = d * jnp.sqrt(r*r - d*d), r*r * jnp.arcsin(d/r)
    fc, fd = c1+c2, d1+d2
    # WRF's center-crossing ELSE is a left-associated four-term REAL sum.
    # Grouping it as fd+fc changes rounding for asymmetric intersections.
    return jnp.where((a > 0.) & (b > 0.), fc-fd,
                     jnp.where((a < 0.) & (b < 0.), fd-fc, (fd+c1)+c2)) * density


def drag_coefficients(speed, diameter, standing, cutout_standing, ws, ct, kw, factor, count=None):
    """Interpolate Cp (NOT power then Cp), exactly as WRF dragcof does."""
    area = jnp.float32(3.1415926535897932384626433) / 4. * diameter**2.
    count = ws.shape[0] if count is None else count
    nb = jnp.clip(jnp.sum((ws <= speed) & (jnp.arange(ws.size)<count))-1, 0, count-2)
    nu = nb + 1
    outside = (speed <= ws[0]) | (speed >= ws[count-1])
    thrust = jnp.where(speed <= ws[0], standing,
             jnp.where(speed >= ws[count-1], cutout_standing,
                       ct[nb]+(ct[nu]-ct[nb])/(ws[nu]-ws[nb])*(speed-ws[nb])))
    fac1 = 1000. / (0.5*1.23*ws[nb]**3.*area)
    fac2 = 1000. / (0.5*1.23*ws[nu]**3.*area)
    cp = kw[nb]*fac1+(kw[nu]*fac2-kw[nb]*fac1)/(ws[nu]-ws[nb])*(speed-ws[nb])
    cp = jnp.where(outside, 0., cp)
    return jnp.maximum(factor*(thrust-cp), 0.), cp, thrust


def dragforce(u, v, z_at_w, dz, qke, du, dv, *, dx, dt, farm):
    """Add WRF Fitch sources to (nz,ny,nx) arrays; return DU,DV,QKE,POWER.

    ``farm`` is a validated immutable host WindFarm. Turbine order, including
    colocated turbines, is preserved. The sparse static loop avoids a dense
    (nt,nz,ny,nx) temporary and does no host/device transfers during stepping.
    """
    density = jnp.float32(1.) / (jnp.float32(dx)*jnp.float32(dx))
    dt = jnp.float32(dt)
    power = jnp.zeros(u.shape[1:], jnp.float32)
    levels = jnp.arange(u.shape[0])
    if not farm.turbines:
        return du,dv,qke,power
    # Compact host constants (nt,max curve rows); one compiled loop irrespective
    # of turbine count. Padding never enters a curve bracket (count is explicit).
    width = max(len(spec[-1]) for _,_,spec in farm.turbines)
    data = [(x,y,spec[0],spec[1],spec[2],len(spec[-1]),
             tuple(spec[-1])+tuple([spec[-1][-1]]*(width-len(spec[-1]))))
            for x,y,spec in farm.turbines]
    columns = tuple(zip(*data))
    xs = tuple(jnp.asarray(c,jnp.int32 if i in (0,1,5) else jnp.float32) for i,c in enumerate(columns))
    def turbine_step(carry, record):
        du,dv,qke,power = carry
        x,y,hub,diameter,standing,count,rows = record
        ws,ct,kw = rows[:,0],rows[:,1],rows[:,2]
        d = jnp.asarray(dz[:, y, x], jnp.float32)
        a, b = jnp.asarray(u[:, y, x], jnp.float32), jnp.asarray(v[:, y, x], jnp.float32)
        # Literal sequential WRF height recurrence, not a parallel prefix sum.
        def edges(h, thick):
            upper = h + thick
            return upper, (h, upper)
        _, (lo, hi) = jax.lax.scan(edges, jnp.float32(0.), d)
        blade_lo, blade_hi = hub-diameter/2., hub+diameter/2.
        bot = jnp.max(jnp.where((blade_lo >= lo) & (blade_lo <= hi), levels, -1))
        top = jnp.min(jnp.where((blade_hi >= lo) & (blade_hi <= hi), levels, d.size))
        active = (levels >= bot) & (levels <= top) & (bot >= 0) & (top < d.size)
        def centers(h, thick):
            mid = h + 0.5*thick
            return mid+0.5*thick, mid
        _, mid = jax.lax.scan(centers, jnp.float32(0.), d)
        ktop = jnp.clip(jnp.sum(mid <= hub), 1, d.size-1)
        kbot = ktop-1
        speed = jnp.sqrt(a**2. + b**2.)
        speedhub = speed[kbot]+((speed[ktop]-speed[kbot])/(mid[ktop]-mid[kbot]))*(hub-mid[kbot])
        tke, cp, thrust = drag_coefficients(speedhub, diameter, standing, ct[count-1], ws, ct, kw,
                                           jnp.float32(farm.tke_factor),count)
        area = jnp.float32(3.1415926535897932384626433)/4.*diameter**2.
        power1 = 0.5*1.23*speedhub**3.*area*cp
        z = jnp.asarray(z_at_w[:, y, x], jnp.float32)
        z1 = jnp.where(active, jnp.maximum(z[:-1]-blade_lo-z[0], 0.), 0.)
        z2 = jnp.where(active, jnp.minimum(z[1:]-blade_lo-z[0], diameter), 0.)
        tarea = turbine_area(z1, z2, diameter, density)
        increments = 0.5*cp*1.23*(speed**3.)*tarea/density
        _, partial = jax.lax.scan(lambda p, inc: (p+inc, p+inc), jnp.float32(0.), increments)
        power2 = partial[-1]
        ec = jnp.where((power1 == 0.) | (power2 == 0.), 1., power1/jnp.where(power2 == 0., 1., power2))
        qke = qke.at[:, y, x].add((speed**3.*tarea*tke*dt/d*ec).astype(qke.dtype))
        du = du.at[:, y, x].add((-.5*a*thrust*speed*tarea/d*ec).astype(du.dtype))
        dv = dv.at[:, y, x].add((-.5*b*thrust*speed*tarea/d*ec).astype(dv.dtype))
        power = power.at[y, x].add(jnp.where(top < d.size, power1, 0.))
        return (du,dv,qke,power),None
    return jax.lax.scan(turbine_step,(du,dv,qke,power),xs)[0]
