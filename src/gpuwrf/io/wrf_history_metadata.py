"""WRF history metadata from input headers and resolved namelist controls."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrf_registry_defaults import WRF_REGISTRY_DEFAULTS
from gpuwrf.io.wrf_header_controls import WRF_HEADER_CONTROLS

# Frozen WRF v4 Registry defaults; per-domain values use the model's namelist resolution.
# Registry.EM_COMMON:2225-2884; registry.stoch:251. MPI task counts describe this GPU process.
_HISTORY_CONTROLS = {
    'AER_ANGEXP_OPT': ('physics', 'integer', True, 1),
    'AER_ANGEXP_VAL': ('physics', 'real', True, 1.3),
    'AER_AOD550_OPT': ('physics', 'integer', True, 1),
    'AER_AOD550_VAL': ('physics', 'real', True, 0.12),
    'AER_ASY_OPT': ('physics', 'integer', True, 1),
    'AER_ASY_VAL': ('physics', 'real', True, 0.9),
    'AER_OPT': ('physics', 'integer', False, 0),
    'AER_SSA_OPT': ('physics', 'integer', True, 1),
    'AER_SSA_VAL': ('physics', 'real', True, 0.85),
    'AER_TYPE': ('physics', 'integer', True, 1),
    'BLDT': ('physics', 'real', True, 0.0),
    'BUCKET_J': ('physics', 'real', False, -1.0),
    'BUCKET_MM': ('physics', 'real', False, -1.0),
    'CLDOVRLP': ('physics', 'integer', False, 2),
    'CUDT': ('physics', 'real', True, 0.0),
    'DFI_OPT': ('dfi_control', 'integer', False, 0),
    'DIFF_6TH_FACTOR': ('dynamics', 'real', True, 0.12),
    'DIFF_6TH_OPT': ('dynamics', 'integer', True, 0),
    'DVEG': ('noah_mp', 'integer', False, 4),
    'FEEDBACK': ('domains', 'integer', False, 1),
    'GRAV_SETTLING': ('physics', 'integer', True, 0),
    'ICLOUD': ('physics', 'integer', False, 1),
    'ICLOUD_CU': ('derived', 'integer', True, 0),
    'IDCOR': ('physics', 'integer', False, 0),
    'ISFFLX': ('physics', 'integer', False, 1),
    'ISFTCFLX': ('physics', 'integer', False, 0),
    'ISHALLOW': ('physics', 'integer', False, 0),
    'MFSHCONV': ('physics', 'integer', True, 1),
    'MOIST_ADV_OPT': ('dynamics', 'integer', True, 1),
    'OBS_NUDGE_OPT': ('fdda', 'integer', True, 0),
    'OPT_ALB': ('noah_mp', 'integer', False, 2),
    'OPT_BTR': ('noah_mp', 'integer', False, 1),
    'OPT_CROP': ('noah_mp', 'integer', False, 0),
    'OPT_CRS': ('noah_mp', 'integer', False, 1),
    'OPT_FRZ': ('noah_mp', 'integer', False, 1),
    'OPT_GLA': ('noah_mp', 'integer', False, 1),
    'OPT_INF': ('noah_mp', 'integer', False, 1),
    'OPT_IRR': ('noah_mp', 'integer', False, 0),
    'OPT_IRRM': ('noah_mp', 'integer', False, 0),
    'OPT_PEDO': ('noah_mp', 'integer', False, 1),
    'OPT_RAD': ('noah_mp', 'integer', False, 3),
    'OPT_RSF': ('noah_mp', 'integer', False, 1),
    'OPT_RUN': ('noah_mp', 'integer', False, 3),
    'OPT_SFC': ('noah_mp', 'integer', False, 1),
    'OPT_SNF': ('noah_mp', 'integer', False, 1),
    'OPT_SOIL': ('noah_mp', 'integer', False, 1),
    'OPT_STC': ('noah_mp', 'integer', False, 1),
    'OPT_TBOT': ('noah_mp', 'integer', False, 2),
    'PREC_ACC_DT': ('physics', 'real', True, 0.0),
    'RADT': ('physics', 'real', True, 0.0),
    'SCALAR_ADV_OPT': ('dynamics', 'integer', True, 1),
    'SCALAR_PBLMIX': ('physics', 'integer', True, 0),
    'SHCU_PHYSICS': ('physics', 'integer', True, 0),
    'SKEBS_ON': ('derived', 'integer', False, 0),
    'SMOOTH_OPTION': ('domains', 'integer', False, 2),
    'SWINT_OPT': ('physics', 'integer', False, 0),
    'SWRAD_SCAT': ('physics', 'real', False, 1.0),
    'TKE_ADV_OPT': ('dynamics', 'integer', True, 1),
    'TRACER_PBLMIX': ('physics', 'integer', True, 1),
    'USE_Q_DIABATIC': ('dynamics', 'integer', False, 0),
    'W_DAMPING': ('dynamics', 'integer', False, 0),
    'YSU_TOPDOWN_PBLMIX': ('physics', 'integer', False, 1),
}


def wrf_history_global_attributes(
    input_path: str | Path,
    namelist: Mapping[str, Mapping[str, Any]],
    domain: str,
    dt_s: float,
) -> dict[str, Any]:
    """Copy static WRF input metadata; derive history-only controls without CPU outputs."""
    with Dataset(input_path, "r") as initial:
        attrs = {name: initial.getncattr(name) for name in initial.ncattrs()
                 if name not in {"USE_TROP_LEVEL", "USE_MAXW_LEVEL"}}
    index = int(domain[1:]) - 1
    controls = dict(_HISTORY_CONTROLS)
    for name, control in WRF_HEADER_CONTROLS.items():
        section, _, _, _ = control
        if name in attrs or name.lower() in namelist.get(section, {}) or section in ('fdda', 'stoch'):
            controls[name] = control
    # wrfinput precedes the forecast namelist. output_wrf.F:625-693 writes
    # the CURRENT configuration, including physics selectors and FDDA controls.
    # Overlay only existing WRF header keys; do not invent a new metadata ABI.
    for name in attrs:
        entry = WRF_REGISTRY_DEFAULTS.get(name.lower())
        if entry is not None:
            section, entries, kind, default = entry
            controls[name] = (section, kind, entries == 'max_domains', default)
    for name, (section, kind, per_domain, default) in controls.items():
        value = namelist.get(section, {}).get(name.lower(), default)
        if isinstance(value, (tuple, list)):
            value = value[min(index, len(value)-1) if per_domain else 0] if value else default
        if value is None:
            value = default
        attrs[name] = np.float32(value) if kind == "real" else np.int32(value)
    # WRF share/module_check_a_mundo.F:1730-1743 resets this QNSE-only control.
    if int(attrs.get("BL_PBL_PHYSICS", 0)) != 4:
        attrs["MFSHCONV"] = np.int32(0)
    def domain_value(section, key, default=0):
        value = namelist.get(section, {}).get(key, default)
        if isinstance(value, (list, tuple)):
            return value[min(index, len(value)-1)] if value else default
        return value
    # check_a_mundo.F:3396-3455 rewrites legacy selectors to NSSL_2MOM.
    # Variant flags retain their distinction; WRF headers record option18.
    if int(attrs.get('MP_PHYSICS',0)) in (17,19,21,22):
        attrs['MP_PHYSICS']=np.int32(18)
    # check_a_mundo.F:988 disables file GHG globally if shortwave is off.
    sw=namelist.get('physics',{}).get('ra_sw_physics',attrs.get('RA_SW_PHYSICS',4))
    sw=sw if isinstance(sw,(list,tuple)) else (sw,)
    max_dom=namelist.get('domains',{}).get('max_dom',len(sw))
    max_dom=max_dom[0] if isinstance(max_dom,(list,tuple)) else max_dom
    if int(attrs.get('GHG_INPUT',0))==1 and any(int(v)==0 for v in sw[:int(max_dom)]):
        attrs['GHG_INPUT']=np.int32(0)
    # module_physics_init.F:1105-1117, not an independent namelist selector.
    cu = int(attrs.get('CU_PHYSICS', 0))
    feedback = bool(domain_value('physics', 'cu_rad_feedback', False))
    attrs['ICLOUD_CU'] = np.int32(2 if cu == 11 or (feedback and cu == 1)
                                 else 1 if feedback and cu in (3, 5, 93) else 0)
    attrs['SKEBS_ON'] = np.int32(bool(domain_value('stoch', 'skebs', 0)))
    # output_wrf.F:733-735 writes TOPO_WIND (after DIFF_6TH_FACTOR) only for topo_wind == 1.
    attrs.pop('TOPO_WIND', None)
    if int(domain_value('physics', 'topo_wind', 0)) == 1:
        items = list(attrs.items())
        at = next((k + 1 for k, (name, _) in enumerate(items) if name == 'DIFF_6TH_FACTOR'), len(items))
        attrs = dict(items[:at] + [('TOPO_WIND', np.int32(1))] + items[at:])
    attrs.update(NTASKS_X=np.int32(1), NTASKS_Y=np.int32(1), NTASKS_TOTAL=np.int32(1))
    attrs["DT"] = np.float32(dt_s)
    return attrs
