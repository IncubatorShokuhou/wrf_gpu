"""WRF BEP+BEM table parameters and ordinary-roof radiation statics."""
from dataclasses import dataclass

import numpy as np

from gpuwrf.physics.urban_bep_static import gauss_plan,radiation_matrix
from gpuwrf.physics.urban_bem_building import room_geometry


def window_albedo(dtype):
    """module_sf_bep_bem.F:albwindow/foncs, p_num=2,q_num=4."""
    T=np.dtype(dtype).type;pi=T(3.141592653)
    g0=T(4)*T(1)*T(1.5)/((T(1)+T(1.5))*(T(1)+T(1.5)))
    a=T(8);b=T(.25)/T(4);c=T(1)-a-b
    alpha=T(5.2)+T(.7)*T(4);delta=T(2)
    gamma=(T(5.26)+T(.06)*T(2))+(T(.73)+T(.04)*T(2))*T(4)
    def integral(n):
        total=T(0)
        for k in range(1,n+1):
            x=pi*T(k)/T(n)
            value=((a*(x**alpha))/(pi**alpha)+(b*(x**delta))/(pi**delta)+(c*(x**gamma))/(pi**gamma))*np.sin(x)
            total=total+(pi/T(n))*value
        return total
    previous=integral(1)
    for n in range(2,10001):
        current=integral(n)
        if abs(previous-current)<T(1.e-7):return T(1)-g0+(g0/T(2))*previous
        previous=current
    raise ValueError('WRF BEM window-albedo quadrature did not converge')


@dataclass(frozen=True,eq=False)
class BemClass:
    bep: object
    category: int
    levels: tuple
    probabilities: np.ndarray
    pwin: object
    window_albedo: object
    room: object
    params: dict
    long_plans: tuple
    short_plans: tuple


def bem_class(bep,category,dtype):
    T=np.dtype(dtype).type;raw=bep.raw
    def value(name):
        a=raw[name]
        return T(a[category] if len(a)>1 else a[0])
    for key in ['PV_FRAC_ROOF','GR_FRAC_ROOF','GR_FLAG']:
        if key in raw and np.any(raw[key]!=0):
            raise NotImplementedError('BEP+BEM requires '+key+'=0; PV/green-roof extension is not qualified')
    if bep.nd!=2:raise NotImplementedError('WRF BEM building has four walls and requires two street directions')
    levels=tuple(i for i in range(1,bep.n) if bep.ss[i]>0.)
    probability=np.asarray([bep.ss[i] for i in levels],dtype)
    pw=value('PWIN');aw=window_albedo(dtype);ewin=T(.9)
    ag,ab,ar=bep.albedo;eg,eb,er=bep.emissivity
    long=tuple(gauss_plan(radiation_matrix(v[0],v[1],v[2],bep.pb,
            T(1)-eb*(T(1)-pw)-ewin*pw,T(1)-eg)) for v in bep.views)
    albavg=pw*aw+(T(1)-pw)*ab
    short=tuple(gauss_plan(radiation_matrix(v[0],v[1],v[2],bep.pb,albavg,ag)) for v in bep.views)
    wall=np.asarray([.02,.02,.02,.02,.02,.02,.02,.01,.005,.0025],dtype)
    ground=np.asarray([.2,.12,.08,.05,.03,.02,.02,.01,.005,.0025],dtype)
    floor=np.full(10,T(.02),dtype)
    cg,cb,cr=bep.capacity;kg,kb,kr=bep.diffusivity
    params={'wall_albedo':ab,'wall_emissivity':eb,'roof_albedo':ar,'roof_emissivity':er,
        'wall_depth':wall,'roof_depth':wall.copy(),'floor_depth':floor,'ground_depth':ground,
        'wall_capacity':np.full(10,cb,dtype),'roof_capacity':np.full(10,cr,dtype),
        'floor_capacity':np.full(10,cb,dtype),'ground_capacity':np.full(10,cg,dtype),
        'wall_conductivity':np.full(11,cb*kb,dtype),'roof_conductivity':np.full(11,cr*kr,dtype),
        'floor_conductivity':np.full(11,cb*kb,dtype),'ground_conductivity':np.full(11,cg*kg,dtype),
        'equipment_profile':np.asarray(raw['HSEQUIP'],dtype),'equipment_scale':value('HSEQUIP_SCALE_FACTOR'),
        'occupants':value('PERFLO'),'beta':value('BETA'),'ac_switch':int(value('SW_COND')),
        'time_on':value('TIME_ON'),'time_off':value('TIME_OFF'),'target_temperature':value('TARGTEMP'),
        'temperature_gap':value('GAPTEMP'),'target_humidity':value('TARGHUM'),'humidity_gap':value('GAPHUM'),
        'cop':value('COP'),'building_ac_fraction':value('BLDAC_FRC'),'cooled_fraction':value('COOLED_FRC')}
    room=room_geometry(bep.building[0],bep.building[1],pw,aw,dtype)
    return BemClass(bep,category,levels,probability,pw,aw,room,params,long,short)
