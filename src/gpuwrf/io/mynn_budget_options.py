"""Backend-dark WRF MYNN history/EDMF option validation."""
from collections.abc import Mapping


def _value(namelist,name,default,domain=None):
    value=(namelist.get('physics',namelist).get(name,default)
           if isinstance(namelist,Mapping) else getattr(namelist,name,default))
    if isinstance(value,(list,tuple)):
        if not value:raise ValueError(f'{name}: empty per-domain selection')
        if domain is None:
            if len(value)!=1:raise ValueError('MYNN budget configuration needs a resolved domain')
            value=value[0]
        else:value=value[min(domain,len(value)-1)]
    return value


def configuration(namelist,domain=None):
    result={}
    for name,default in (('tke_budget',0),('bl_mynn_output',0),('bl_mynn_edmf',1)):
        value=_value(namelist,name,default,domain)
        if value not in (0,1):raise ValueError(f'{name} supports only 0/1')
        result[name]=int(value)
    return result


def validate_options(namelist,domain=None):
    config=configuration(namelist,domain)
    active=bool(config['tke_budget'] or config['bl_mynn_output'] or not config['bl_mynn_edmf'])
    if active and int(_value(namelist,'bl_pbl_physics',5,domain))!=5:
        raise ValueError('tke_budget/bl_mynn_output/bl_mynn_edmf=0 require bl_pbl_physics=5 (MYNN)')
    # The master EDMF-off switch removes every mass-flux path. A separate
    # momentum-only disable is not currently wired in the production kernel.
    mom=_value(namelist,'bl_mynn_edmf_mom',1,domain)
    if mom not in (0,1):raise ValueError('bl_mynn_edmf_mom supports only 0/1')
    if config['bl_mynn_edmf'] and not mom:
        raise ValueError('bl_mynn_edmf_mom=0 with bl_mynn_edmf=1 is not supported')
    return config
