"""Advisory initial turbine drag time scale; never a stability guarantee."""
from collections import Counter
import math


def density_advice(farm, *, dx, dt, min_rotor_dz, max_speed):
    if any(not math.isfinite(float(x)) or x<=0 for x in (dx,dt,min_rotor_dz)) or not math.isfinite(max_speed) or max_speed<0:
        raise ValueError('windfarm preflight requires positive dx/dt/dz and finite nonnegative speed')
    cells = Counter((x,y) for x,y,_ in farm.turbines)
    sums = Counter()
    for x,y,spec in farm.turbines:
        _,diameter,standing,_,rows = spec
        ct = max(standing,max(row[1] for row in rows))
        sums[x,y] += .5*ct*math.pi/4*diameter**2/(dx*dx)*max_speed*dt/min_rotor_dz
    return {'turbines':len(farm.turbines),'max_turbines_per_cell':max(cells.values(),default=0),
            'max_density_m_minus2':max(cells.values(),default=0)/(dx*dx),
            'initial_drag_dt_bound':max(sums.values(),default=0.),
            'warning':max(sums.values(),default=0.)>.5,
            'scope':'advisory full rotor area and maximum tabulated Ct, initial max speed/min dz; '
                    'excludes WRF energy-normalization ec, future flow and PBL/acoustic stability'}
