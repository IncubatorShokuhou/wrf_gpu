"""Host-only E78 admission for device-resident urban forcing inputs."""
from collections.abc import Mapping
import numpy as np

KEYS = ('required','ahe','diurnal','morphology','clock')


def configuration(namelist):
    if isinstance(namelist,Mapping) and set(namelist)==set(KEYS):
        return dict(namelist)
    bundle = namelist.get('urban_bundle') if isinstance(namelist,Mapping) else getattr(namelist,'urban_bundle',None)
    if bundle is None or getattr(bundle,"option",1) in (2,3):
        return dict.fromkeys(KEYS,False)
    timed=bool(bundle.ahe_opt or bundle.table.ahoption or bundle.table.alhoption)
    return {'required':bool(bundle.requires_inputs),'ahe':bool(bundle.ahe_opt),
        'diurnal':bool(bundle.table.ahoption or bundle.table.alhoption),
        'morphology':bool(getattr(bundle,'has_morphology',False)),
        'clock':timed and bundle.start is None}


def validate_urban_inputs(group,config,version,shape):
    """Never instantiate JAX/device arrays while admitting an on-disk snapshot."""
    cfg=configuration(config)
    if any(type(cfg[k]) is not bool for k in KEYS) or (any(cfg[k] for k in KEYS[1:]) and not cfg['required']):
        raise ValueError('restart urban input configuration invalid (E78)')
    inputs=getattr(group,'inputs',None)
    if not cfg['required'] and inputs is None:
        return
    if not cfg['required']:
        raise ValueError('restart urban inputs are active without matching configuration (E78)')
    if version!=1 or inputs is None or tuple(getattr(inputs,'_fields',())) != ('morphology','longitude','ahe','hourly','clock'):
        raise ValueError('restart urban LCZ/AHE/morphology inputs schema missing or incompatible (E78)')
    shape=tuple(shape)
    def check(a,expected,name):
        if a is None or np.shape(a)!=expected or np.asarray(a).dtype!=np.float32 or not np.isfinite(np.asarray(a)).all():
            raise ValueError(f'restart urban {name} shape/dtype/nonfinite mismatch (E78)')
    if cfg['clock'] or inputs.clock is not None:
        check(inputs.clock,(4,),'clock')
        day,hour,minute,month=np.asarray(inputs.clock)
        if not (1<=day<=366 and day==int(day) and 0<=hour<24 and 0<=minute<1440 and 1<=month<=12 and month==int(month)):
            raise ValueError('restart urban clock anchors outside calendar ranges (E78)')
    if cfg['diurnal']:
        check(inputs.longitude,shape,'longitude')
    if cfg['morphology']:
        names=('zr','r','rw','hgt','zdc','z0c','z0hc','z0r','svf')
        if tuple(getattr(inputs.morphology,'_fields',()))!=names:
            raise ValueError('restart urban morphology fields missing (E78)')
        for key in names: check(getattr(inputs.morphology,key),shape,key)
    if cfg['ahe']:
        if inputs.hourly is not None:
            s=np.shape(inputs.hourly)
            if len(s)!=3 or s[0]<1:
                raise ValueError('restart hourly urban AHE dimensions invalid (E78)')
            check(inputs.hourly,(s[0],)+shape,'hourly AHE')
        else:
            check(inputs.ahe,(288,)+shape,'AHE')
