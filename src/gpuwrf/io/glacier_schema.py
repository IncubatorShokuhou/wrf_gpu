"""Fail-closed restart schema for the glacier's previous QSNOWXY."""
from collections.abc import Mapping
import numpy as np

VERSION=1


def required(namelist):
    static=(namelist.get('noahmp_static') if isinstance(namelist,Mapping) else getattr(namelist,'noahmp_static',None))
    if static is None:return False
    if getattr(static,'glacier',None) is not None:return True
    vegetation=getattr(static,'ivgtyp',None)
    parameters=getattr(static,'parameters',None)
    return vegetation is not None and bool(np.any(np.asarray(vegetation)==int(getattr(parameters,'isice',15))))


def field_order(land):
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
    return NoahMPLandState.__slots__ if getattr(land,'glacier_qsnow',None) is not None else NoahMPLandState.__slots__[:-1]


def validate(fields,order,version,*,namelist=None,active=None,shape=None):
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
    expected=NoahMPLandState.__slots__
    if active is None:active=required(namelist)
    if fields is None:
        if active or version is not None:raise ValueError('active glacier restart has no Noah-MP land carry (E78)')
        return
    if tuple(order) not in (expected,expected[:-1]) or set(fields)!=set(order):
        raise ValueError('restart Noah-MP land field order/fields differ (E78)')
    q=fields.get('glacier_qsnow')
    if active or q is not None or version is not None:
        if version!=VERSION or q is None:
            raise ValueError('active glacier restart lacks versioned previous QSNOWXY (E78)')
        a=np.asarray(q);target=np.shape(fields['sneqv']) if shape is None else tuple(shape)
        if a.shape!=target or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all():
            raise ValueError('glacier QSNOWXY restart shape/dtype/nonfinite mismatch (E78)')


def marker(land):
    return VERSION if land is not None and getattr(land,'glacier_qsnow',None) is not None else None
