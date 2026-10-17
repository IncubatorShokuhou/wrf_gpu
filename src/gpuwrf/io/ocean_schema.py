"""Admission of ocean/USTM carries at restart boundaries (E78)."""
SCHEMA_VERSION = 1
FIELDS = ('tml','t0ml','hml','h0ml','huml','hvml','tmoml')
CONFIG_KEYS = ('sf_ocean_physics','oml_hml0','oml_gamma','oml_relaxation_time','isftcflx')


def configuration(namelist):
    from collections.abc import Mapping
    defaults = (0,50.,.14,0.,0)
    if isinstance(namelist,Mapping):
        namelist=namelist.get('physics',namelist)
        return {k:namelist.get(k,v) for k,v in zip(CONFIG_KEYS,defaults,strict=True)}
    return {k:getattr(namelist,k,v) for k,v in zip(CONFIG_KEYS,defaults,strict=True)}


def validate(ocean, ustm, namelist, version, *, shape=None, saved=None, history=None, require_history=False):
    import numpy as np
    cfg = configuration(namelist)
    active = cfg['sf_ocean_physics']==1 or cfg['isftcflx']>0
    if not active and ocean is None and ustm is None:
        return
    if version != SCHEMA_VERSION:
        raise ValueError('ocean/hurricane restart schema missing (E78)')
    if saved is not None and configuration(saved)!=cfg:
        raise ValueError('ocean/hurricane restart configuration differs (E78)')
    if (ocean is not None)!=(cfg['sf_ocean_physics']==1):
        raise ValueError('ocean restart carry/selection disagree (E78)')
    # Only MYNN uses USTM in its dissipative heating; MM5 passes it unused.
    need_ustm = cfg['isftcflx']>0
    if (ustm is not None)!=need_ustm:
        raise ValueError('hurricane USTM restart carry missing/unexpected (E78)')
    if ocean is not None and tuple(getattr(ocean,'_fields',()))!=FIELDS:
        raise ValueError('ocean restart fields missing/differ (E78)')
    if ocean is not None and require_history:
        lh=np.asarray(getattr(history,'lh',None))
        if lh.dtype!=np.float32 or lh.ndim!=2 or (shape is not None and lh.shape!=tuple(shape)) or not np.isfinite(lh).all():
            raise ValueError('ocean restart current surface LH missing/invalid (E78)')
    values = (() if ocean is None else tuple(ocean)) + (() if ustm is None else (ustm,))
    for value in values:
        a=np.asarray(value)
        if a.dtype!=np.float32 or a.ndim!=2 or (shape is not None and a.shape!=tuple(shape)) or not np.isfinite(a).all():
            raise ValueError('ocean/USTM restart shape/dtype/nonfinite differs (E78)')
