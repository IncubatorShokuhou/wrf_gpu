"""Host-only Noah-MP restart-table identity, outside traced physics imports.

The canonical digest and validation are unchanged. Keeping this helper with
restart IO prevents AOT orchestration from entering the model source closure.
"""
import re


def bound_table_identity(namelist):
    """Resolved nondefault tables/maps, excluding unused cold prognostic seeds."""
    from collections.abc import Mapping
    get=namelist.get if isinstance(namelist,Mapping) else lambda key,default=None:getattr(namelist,key,default)
    if get('noahmp_options') is None:return None
    static=get('noahmp_static');carbon=get('noahmp_carbon_static');irrigation=get('noahmp_irrigation_static')
    from gpuwrf.runtime.aot_cheap_key import canonical_digest
    return canonical_digest((getattr(static,'parameters',None),get('noahmp_energy_params'),get('noahmp_rad_params'),
        getattr(carbon,'parameters',None),None if irrigation is None else irrigation._replace(initial=None)))



def validate_bound_table_restart(recorded,namelist=None):
    if recorded is not None and (not isinstance(recorded,str) or re.fullmatch('[0-9a-f]{64}',recorded) is None):
        raise ValueError('Noah-MP bound table restart identity is malformed (E78)')
    if namelist is not None and recorded!=bound_table_identity(namelist):
        raise ValueError('Noah-MP bound table/map values differ from requested restart (E78)')
