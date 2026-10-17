"""Backend-dark selection of the independent WRF glacier model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GlacierStatic:
    indices: tuple[int, ...]
    natural_indices: tuple[int, ...]
    shape: tuple[int, int]
    opt_gla: int = 1
    opt_alb: int = 2
    opt_snf: int = 1
    opt_tbot: int = 2
    opt_stc: int = 1


def glacier_options(namelist):
    group=namelist.get('noah_mp') or {}
    defaults=dict(opt_gla=1,opt_alb=2,opt_snf=1,opt_tbot=2,opt_stc=1)
    out={}
    for name,default in defaults.items():
        value=group.get(name,default)
        if isinstance(value,(tuple,list)):value=value[0]
        if isinstance(value,bool) or not isinstance(value,int):
            raise ValueError(f'&noah_mp {name} must be an integer')
        out[name]=value
    supported={'opt_gla':(1,2),'opt_alb':(1,2),'opt_snf':(1,2,3,4,5),'opt_tbot':(1,2),'opt_stc':(1,2,3)}
    for name,choices in supported.items():
        if out[name] not in choices:
            raise ValueError(f'&noah_mp {name}: supported values {choices}')
    return out


def glacier_refusals(namelist):
    try:
        glacier_options(namelist)
    except ValueError as exc:
        return [str(exc)]
    return []
