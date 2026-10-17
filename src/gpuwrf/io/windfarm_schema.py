"""Fitch restart guard: selection, turbine-table identity and held POWER."""
import numpy as np


def configuration(namelist):
    get = namelist.get if isinstance(namelist,dict) else lambda k,d=None:getattr(namelist,k,d)
    farm = get('windfarm')
    return {'windfarm_opt':int(get('windfarm_opt',0) or 0),
            'fingerprint':getattr(farm,'fingerprint',''),
            'tke_factor':float(get('windfarm_tke_factor',.25))}


def validate_restart(power, saved, shape, *, requested=None):
    config = configuration(requested) if requested is not None else saved
    if config is None:
        if power is not None:
            raise ValueError('Fitch restart configuration is missing (E78)')
        return
    if (not isinstance(config,dict) or set(config)!={'windfarm_opt','fingerprint','tke_factor'}
            or type(config['windfarm_opt']) is not int or config['windfarm_opt'] not in (0,1)
            or not isinstance(config['fingerprint'],str)
            or not isinstance(config['tke_factor'],(int,float))
            or not np.isfinite(config['tke_factor']) or config['tke_factor']<0):
        raise ValueError('Fitch restart configuration is invalid (E78)')
    active = config.get('windfarm_opt',0) == 1
    if saved is not None and config != saved:
        raise ValueError('Fitch restart turbine configuration mismatch (E78)')
    if active:
        if saved is None or not config.get('fingerprint') or power is None:
            raise ValueError('Fitch restart requires turbine identity and POWER (E78)')
        a = np.asarray(power)
        if a.shape != tuple(shape) or a.dtype != np.float32 or not np.isfinite(a).all() or (a<0).any():
            raise ValueError('Fitch restart POWER must be finite nonnegative REAL surface array (E78)')
    elif power is not None:
        raise ValueError('Fitch POWER carry present with windfarm_opt=0 (E78)')
