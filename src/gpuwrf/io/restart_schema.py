"""Host-side admission of optional held physics at restart boundaries."""
from collections.abc import Mapping

CAMUW_SCHEMA_VERSION = 1
TEMF_SCHEMA_VERSION = 1
CAMUW_FIELDS = ("kvm3d", "kvh3d", "tauresx2d", "tauresy2d", "tke_pbl")
EFFECTIVE_RADIUS_FIELDS = ("re_cloud", "re_ice", "re_snow")
LAKE_SCHEMA_VERSION = 1
QNSE_SCHEMA_VERSION = 1
GODDARD_RADIATION_SCHEMA_VERSION = 2


def history_field_required(field, namelist):
    """Resolve optional held surface outputs from their producing scheme."""
    if field in ('u10e', 'v10e'):
        return physics_option(namelist, 'sf_sfclay_physics') in (2, 4)
    return field != 'hfx'  # State owns HFX when this history slot is absent.


def validate_history_winds(history, namelist, *, shape=None):
    """MYJSFC/QNSESFC must retain both REAL equivalent-roughness winds (E78)."""
    if not history_field_required('u10e', namelist):
        return
    import numpy as np
    if shape is None:
        shape = np.shape(getattr(history, 't2', None))
    for field in ('u10e', 'v10e'):
        value = getattr(history, field, None)
        if (value is None or len(shape) != 2 or np.shape(value) != tuple(shape)
                or np.asarray(value).dtype != np.float32 or not np.isfinite(value).all()):
            scheme = 'QNSE' if physics_option(namelist, 'sf_sfclay_physics') == 4 else 'MYJSFC'
            raise ValueError(f'restart {scheme} missing/invalid history.{field} (E78)')


def validate_qnse(group,namelist,schema_version,*,history=None,expected_shape=None):
    # Every restart API calls this surface admission, including inactive QNSE.
    validate_history_winds(history, namelist,
                           shape=None if expected_shape is None else expected_shape[-2:])
    # The QnseCarry group is the Janjic-family surface memory: QNSESFC (sf=4) and
    # its parent MYJSFC (sf=2) share THZ0/QZ0/UZ0/VZ0/AKHS/AKMS/CT/ELFLX.
    qnse=physics_option(namelist,'bl_pbl_physics')==4 or physics_option(namelist,'sf_sfclay_physics')==4
    myj=physics_option(namelist,'sf_sfclay_physics')==2
    active=qnse or myj
    if group is None and not active:return
    from gpuwrf.coupling.qnse_adapter import QnseCarry
    if schema_version!=QNSE_SCHEMA_VERSION or not isinstance(group,QnseCarry):
        if myj and not qnse:
            raise ValueError('restart MYJSFC is missing/version-mismatched surface memory (E78)')
        raise ValueError('restart QNSE is missing/version-mismatched surface/MF memory (E78)')
    import numpy as np
    surface_shape=np.shape(group.akhs)
    profile_shape=np.shape(group.rc_mf)
    if len(surface_shape)!=2 or len(profile_shape)!=3 or profile_shape[0]<3 or profile_shape[1:]!=surface_shape:
        raise ValueError('restart QNSE carry dimensions differ (E78)')
    if expected_shape is not None and tuple(profile_shape)!=tuple(expected_shape):
        raise ValueError('restart QNSE carry differs from the actual State domain (E78)')
    for name,value in zip(group._fields,group,strict=True):
        shape=profile_shape if name in ('rc_mf','exch_h','exch_m') else surface_shape
        if np.shape(value)!=shape or np.asarray(value).dtype!=np.float32 or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'restart QNSE {name} missing/shape/dtype/nonfinite differs (E78)')
    from gpuwrf.runtime.operational_state import HistoryDiagnostics
    if not isinstance(history,HistoryDiagnostics):
        raise ValueError('restart QNSE surface output cache is missing (E78)')
    for name in history._fields:
        value=getattr(history,name)
        if value is None and not history_field_required(name,namelist):continue
        if np.shape(value)!=surface_shape or np.asarray(value).dtype!=np.float32 or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'restart QNSE output {name} missing/shape/dtype/nonfinite differs (E78)')


def validate_sas_held(group, namelist, rng_key=None, kf_carry=None):
    """SAS cadence reuses every held rate after restart, including RU/RV in 14/96."""
    option = physics_option(namelist, 'cu_physics')
    if option not in (14, 94, 95, 96, 99):
        return
    import numpy as np
    count = 7 if option in (94,95,99) else 9
    if not isinstance(group, (tuple, list)) or len(group) != count:
        raise ValueError(f'restart cu_physics={option} missing {count} held SAS rates (E78)')
    shape = np.shape(group[0])
    grid = getattr(namelist, 'grid', None)
    if len(shape) != 3 or (grid is not None and shape != (grid.nz, grid.ny, grid.nx)):
        raise ValueError('restart SAS held-rate domain differs (E78)')
    for i, value in enumerate(group):
        if (np.shape(value) != (shape[1:] if i == 6 else shape)
                or np.asarray(value).dtype != np.float32 or not np.isfinite(np.asarray(value)).all()):
            raise ValueError(f'restart SAS held rate {i} shape/dtype/nonfinite differs (E78)')
    if option == 99:
        if not isinstance(kf_carry,(tuple,list)) or len(kf_carry)!=2:
            raise ValueError('restart old KF W0AVG/NCA pair missing (E78)')
        for name,value,expected in zip(('W0AVG','NCA'),kf_carry,(shape,shape[1:]),strict=True):
            if (np.shape(value)!=expected or np.asarray(value).dtype!=np.float32
                    or not np.isfinite(np.asarray(value)).all()):
                raise ValueError(f'restart old KF {name} shape/dtype/nonfinite differs (E78)')
    if option == 95:
        from gpuwrf.physics.osas_rng import DEFAULT_SEED
        domain=physics_option(namelist,'osas_domain_id',1)
        if not 1 <= domain <= 0xffffffff:
            raise ValueError('restart OSAS domain key invalid (E78)')
        if (np.shape(rng_key)!=(2,) or np.asarray(rng_key).dtype!=np.uint32
                or not np.array_equal(rng_key,np.array([DEFAULT_SEED,domain],np.uint32))):
            raise ValueError('restart OSAS fixed-seed/domain keys missing or differ (E78)')


def validate_lake(group,namelist,schema_version):
    if group is None and physics_option(namelist,'sf_lake_physics') != 1:
        return
    from gpuwrf.io.lake_schema import LAKE_FIELDS
    if schema_version != LAKE_SCHEMA_VERSION or group is None:
        raise ValueError('restart lake schema/carry missing (E78)')
    if tuple(getattr(group,'_fields',())) != ('columns','surface','energy_residual','background'):
        raise ValueError('restart lake carry fields differ (E78)')
    if tuple(getattr(group.columns,'_fields',())) != tuple(name for name,_,_ in LAKE_FIELDS):
        raise ValueError('restart lake column fields missing/differ (E78)')
    if tuple(getattr(group.surface,'_fields',())) != ('hfx','lh','grdflx','tsk','qfx','t2','th2','q2','albedo'):
        raise ValueError('restart lake held surface fields missing/differ (E78)')
    if any(value is None for value in (*group.columns,*group.surface,group.energy_residual,group.background)):
        raise ValueError('restart lake leaf missing (E78)')
    import numpy as np
    count=np.shape(group.surface.tsk)[0]
    for (name,_,levels),value in zip(LAKE_FIELDS,group.columns,strict=True):
        expected=(count,levels) if levels else (count,)
        if np.shape(value)!=expected or np.asarray(value).dtype!=np.float32 or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'restart lake {name} shape/dtype/nonfinite differs (E78)')
    for name,value in zip(group.surface._fields,group.surface,strict=True):
        if np.shape(value)!=(count,) or np.asarray(value).dtype!=np.float32 or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'restart lake held {name} shape/dtype/nonfinite differs (E78)')
    if np.shape(group.energy_residual)!=(count,) or np.asarray(group.energy_residual).dtype!=np.float32 or not np.isfinite(np.asarray(group.energy_residual)).all():
        raise ValueError('restart lake energy diagnostic invalid (E78)')
    if len(np.shape(group.background))!=3 or np.shape(group.background)[0]!=2 or np.asarray(group.background).dtype!=np.float32 or not np.isfinite(np.asarray(group.background)).all():
        raise ValueError('restart lake background invalid (E78)')
    bundle=getattr(namelist,'lake_bundle',None)
    if bundle is not None and (count!=len(bundle.indices) or np.shape(group.background)!=(2,*bundle.shape)):
        raise ValueError('restart lake partition differs from current domain (E78)')


def physics_option(namelist, name, default=0):
    if isinstance(namelist, Mapping):
        return int(namelist.get("physics", namelist).get(name, default))
    return int(getattr(namelist, name, default))


def effective_radius_fields(namelist):
    return ("re_cloud", "re_ice") if physics_option(namelist, "mp_physics", 8) in (50,51,52,53,55) else EFFECTIVE_RADIUS_FIELDS


def effective_radii_required(namelist):
    # Registry allocation alone does not activate radiation mediation. In
    # particular, supported mp40/aercu_opt=0 has pristine has_reqc/i/s=000.
    from gpuwrf.physics.rrtmg_mp_re import mp_re_active
    return physics_option(namelist, "mp_physics", 8) in (50,51,52,53,55) or mp_re_active(namelist)


def validate_goddard_radiation(group,namelist=None,schema_version=None,*,state_shape=None):
    """A selected Goddard restart retains its full flux/rate counter contract."""
    active=5 in (physics_option(namelist,'ra_lw_physics',4),physics_option(namelist,'ra_sw_physics',4))
    own=group is not None and type(group).__name__=='GoddardRadiationDiagnostics'
    if not active and not own and schema_version is None:return
    if namelist is not None and (physics_option(namelist,'ra_lw_physics',4),
                                physics_option(namelist,'ra_sw_physics',4))!=(5,5):
        raise ValueError('restart Goddard bound radiation selection differs (E78)')
    from gpuwrf.coupling.goddard_radiation import GoddardRadiationDiagnostics
    import numpy as np
    if schema_version!=GODDARD_RADIATION_SCHEMA_VERSION or not isinstance(group,GoddardRadiationDiagnostics):
        raise ValueError('restart Goddard schema/held radiation missing (E78)')
    surface_shape=np.shape(group.glw)
    cloud_shape=np.shape(group.cloud_fraction)
    grid=getattr(namelist,'grid',None)
    if (len(surface_shape)!=2 or len(cloud_shape)!=3 or cloud_shape[1:]!=surface_shape
            or (grid is not None and (surface_shape!=(grid.ny,grid.nx)
                                    or cloud_shape!=(grid.nz,grid.ny,grid.nx)))
            or (state_shape is not None and cloud_shape!=tuple(state_shape))):
        raise ValueError('restart Goddard domain shape mismatch (E78)')
    for name in GoddardRadiationDiagnostics._fields:
        if '_clear_' in name:
            if getattr(group,name) is not None:
                raise ValueError('restart Goddard unimplemented clear-sky field present (E78)')
            continue
        value=getattr(group,name)
        if value is None or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'restart Goddard {name} missing/nonfinite (E78)')
        expected_shape=((surface_shape[0]*surface_shape[1],cloud_shape[0],6) if name=='radius_inputs' else
                        () if name=='sw_heating_negative_count' else
                        cloud_shape if name=='cloud_fraction' else surface_shape)
        expected_dtype=np.uint32 if name=='sw_heating_negative_count' else np.int32 if name=='shadow_mask' else np.float32
        if np.shape(value)!=expected_shape or np.asarray(value).dtype!=expected_dtype:
            raise ValueError(f'restart Goddard {name} shape/dtype mismatch (E78)')
    count=np.asarray(group.sw_heating_negative_count)
    if count.shape!=() or count.dtype!=np.uint32:
        raise ValueError('restart Goddard pre-mask counter shape/dtype mismatch (E78)')


def validate_held_state(fields, namelist):
    mp = physics_option(namelist, "mp_physics", 8)
    # These are held producer outputs, never reconstructed at resume/output.
    required_profiles = (("rimi",) if mp == 13 else ()) + (
        ("exch_tke",) if physics_option(namelist, "bl_pbl_physics", 5) == 12 else ())
    if required_profiles or mp == 28:
        import numpy as np
        shape = np.shape(fields.get("qv"))
        for name in (*required_profiles, *(("nwfa2d", "nifa2d") if mp == 28 else ())):
            value = fields.get(name)
            expected = shape[-2:] if name.endswith("2d") else shape
            if (value is None or np.shape(value) != expected
                    or np.asarray(value).dtype != np.float32 or not np.isfinite(np.asarray(value)).all()):
                raise ValueError(f"restart held history {name} missing/shape/dtype/nonfinite differs (E78)")
    if mp == 56:
        import numpy as np
        from gpuwrf.physics.ntu.registry import output_mapping, SURFACE_LEAVES
        required = (*output_mapping().values(), *SURFACE_LEAVES, "hail_acc")
        shape = np.shape(fields.get("qv"))
        for name in required:
            value = fields.get(name)
            expected = shape[-2:] if name in (*SURFACE_LEAVES, "hail_acc") else shape
            allowed = (np.dtype("float32"),) if name.startswith("ntu_") else (np.dtype("float32"), np.dtype("float64"))
            if (value is None or len(shape) != 3 or np.shape(value) != expected
                    or np.asarray(value).dtype not in allowed or not np.isfinite(np.asarray(value)).all()):
                raise ValueError(f"restart NTU {name} missing/shape/dtype/nonfinite differs (E78)")
    from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_leaves
    import numpy as np
    shape = np.shape(fields.get("qv"))
    for name in ice_diagnostic_leaves(mp):
        value = fields.get(name)
        if (value is None or len(shape) != 3 or np.shape(value) != shape
                or np.asarray(value).dtype != np.dtype("float32")
                or not np.isfinite(np.asarray(value)).all()):
            raise ValueError(f"restart held ice diagnostic {name} missing/shape/dtype/nonfinite differs (E78)")
    if mp == 55:
        import numpy as np
        from gpuwrf.physics.jensen_ishmael.registry import input_variables
        required = (*input_variables(), "Ni", "Nr", "re_cloud", "re_ice")
        shape = np.shape(fields.get("qv"))
        for name in required:
            value = fields.get(name)
            allowed = (np.dtype('float32'), np.dtype('float64')) if name in ('Ni', 'Nr') else (np.dtype('float32'),)
            if (value is None or len(shape) != 3 or np.shape(value) != shape
                    or np.asarray(value).dtype not in allowed
                    or not np.isfinite(np.asarray(value)).all()):
                raise ValueError(f"restart Jensen {name} missing/shape/dtype/nonfinite differs (E78)")
    if mp in (50,51,52,53):
        from gpuwrf.contracts.state import conditional_state_leaves_for_mp
        required = conditional_state_leaves_for_mp(mp) + ("Ni","Nr") + (("Nc",) if mp != 50 else ())
        missing = [name for name in required if fields.get(name) is None]
        if missing:
            raise ValueError(f"restart P3 mp_physics={mp} is missing prognostic/old-physics carry (E78): {missing}")
    numbers = {10: ("Ni", "Nr", "Ns", "Ng"), 14: ("Nc", "Nr", "Nn"),
               16: ("Nc", "Nr", "Nn"), 26: ("Nc", "Nr", "Nn"),
               9: ("qh", "Nc", "Nr", "Ni", "Ns", "Ng", "Nh", "hail_acc")}.get(mp, ())
    if any(fields.get(name) is None for name in numbers):
        raise ValueError(f"restart mp_physics={mp} is missing prognostic number scalars {numbers} (E78)")
    if mp == 9:
        import numpy as np
        shape=np.shape(fields['qv'])
        for name in numbers:
            value=fields[name]
            expected=shape[-2:] if name=='hail_acc' else shape
            if np.shape(value)!=expected or np.dtype(value.dtype) not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError(f'restart Milbrandt {name} shape/dtype differs (E78)')
    ferrier = {5: ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr"),
               15: ("qrimef", "ferrier_sr")}.get(mp, ())
    if any(fields.get(name) is None for name in ferrier):
        raise ValueError(f"restart mp_physics={mp} is missing Ferrier carry {ferrier} (E78)")
    if mp == 7:
        from gpuwrf.contracts.precision import GODDARD4ICE_CARRY_LEAVES
        required = ("qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES)
        if any(fields.get(name) is None for name in required):
            raise ValueError("restart mp_physics=7 is missing Goddard4ice held carry (E78)")
    if mp in (5, 7, 15):
        import numpy as np
        required = required if mp == 7 else ferrier
        shape = np.shape(fields.get("qv"))
        if len(shape) != 3:
            raise ValueError(f"restart mp_physics={mp} has no mass-grid qv shape (E78)")
        for name in required:
            value = fields[name]
            expected = shape[-2:] if name in ("ferrier_sr", "goddard_sr", "hail_acc") else shape
            # Held WRF REAL fields stay fp32, including force-fp64 runs;
            # qh/hail_acc also support the retained wide prognostic mode.
            allowed = (np.dtype("float32"), np.dtype("float64")) if name in ("qh", "hail_acc") else (np.dtype("float32"),)
            if np.shape(value) != expected or np.dtype(value.dtype) not in allowed:
                raise ValueError(f"restart mp_physics={mp} {name} shape/dtype differs (E78)")
    if mp == 38:
        import numpy as np
        from gpuwrf.contracts.precision import THOMPSON_HAIL_SURFACE_LEAVES
        required = ("qvolg", "nwfa", "nifa", "Nc", "Ng", "Ni", "Nr", *THOMPSON_HAIL_SURFACE_LEAVES)
        if any(fields.get(name) is None for name in required):
            raise ValueError("restart mp_physics=38 is missing Thompson-hail carry/emissions (E78)")
        shape = np.shape(fields.get("qv"))
        if len(shape) != 3:
            raise ValueError("restart mp_physics=38 has no mass-grid qv shape (E78)")
        for name in required:
            value = fields[name]
            surface = name in THOMPSON_HAIL_SURFACE_LEAVES
            expected = shape[-2:] if surface else shape
            allowed = (np.dtype("float32"),) if surface else (np.dtype("float32"), np.dtype("float64"))
            if np.shape(value) != expected or np.dtype(value.dtype) not in allowed:
                raise ValueError(f"restart mp_physics=38 {name} shape/dtype differs (E78)")
    from gpuwrf.nesting.scalar_species import required_extra_boundary_records
    missing = [name for name in required_extra_boundary_records(namelist) if fields.get(name) is None]
    if missing:
        raise ValueError(f"restart is missing live-child extra scalar boundary records (E78): {missing}")
    from gpuwrf.contracts.state import mynn_sfc_wspd_enabled
    if mynn_sfc_wspd_enabled() and fields.get("sfc_wspd") is None:
        raise ValueError("restart is missing held MYNN sfc_wspd (E78)")
    active = effective_radii_required(namelist) or any(
        fields.get(name) is not None for name in EFFECTIVE_RADIUS_FIELDS)
    radius_fields = effective_radius_fields(namelist)
    if active and any(fields.get(name) is None for name in radius_fields):
        raise ValueError("restart is missing held microphysics effective radii (E78)")
    if physics_option(namelist, "km_opt") == 5:
        from gpuwrf.contracts.precision import SMS_CARRY_LEAVES
        if fields.get("tke") is None or any(fields.get(n) is None for n in SMS_CARRY_LEAVES):
            raise ValueError("km_opt=5 restart is missing SMS surface/nonlocal carry or TKE (E78)")


def validate_camuw(group, namelist, schema_version):
    if group is None and physics_option(namelist, "bl_pbl_physics") != 9:
        return
    if schema_version != CAMUW_SCHEMA_VERSION:
        raise ValueError("restart CAM-UW schema predates held PBL carry (E78)")
    if group is None or tuple(getattr(group, "_fields", ())) != CAMUW_FIELDS:
        raise ValueError("restart CAM-UW is missing held PBL carry (E78)")
    if any(getattr(group, name) is None for name in CAMUW_FIELDS):
        raise ValueError("restart CAM-UW is missing held PBL leaves (E78)")


def validate_temf(group, namelist, schema_version):
    if group is None and physics_option(namelist,'bl_pbl_physics') != 10:
        return
    if schema_version != TEMF_SCHEMA_VERSION or group is None:
        raise ValueError('restart TEMF schema/carry missing (E78)')
    from gpuwrf.coupling.temf import TemfCarry,TEMF_PROFILES,SURFACE_HELD
    import numpy as np
    if not isinstance(group,TemfCarry) or set(group.profiles) != set(TEMF_PROFILES) or set(group.surface) != set(SURFACE_HELD):
        raise ValueError('restart TEMF Registry/held fields differ (E78)')
    shape=np.shape(group.profiles['te_temf'])
    if len(shape)!=2:
        raise ValueError('restart TEMF column shape differs (E78)')
    grid=getattr(namelist,'grid',None)
    if grid is not None and shape!=(grid.ny*grid.nx,grid.nz):
        raise ValueError('restart TEMF column/domain shape differs (E78)')
    for k,v in (*group.profiles.items(),*group.surface.items()):
        expected = shape if k in TEMF_PROFILES else (shape[0],)
        if np.shape(v)!=expected or np.asarray(v).dtype!=(np.int32 if k=='kpbl' else np.float32) or not np.isfinite(np.asarray(v)).all():
            raise ValueError(f'restart TEMF {k} shape/dtype/nonfinite differs (E78)')


def validate_spectral_fdda(group, namelist, *, required=False):
    """Spectral input selection must never resume with a PSU/missing carry."""
    from gpuwrf.assimilation.fdda_surface import validate_surface_restart
    validate_surface_restart(group,namelist)
    validate_grid_fdda(group, namelist)
    from dataclasses import fields
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs, SpectralFddaCarry
    active = required or isinstance(getattr(namelist, "data_assimilation", None), SpectralFddaInputs)
    if not active and not isinstance(group, SpectralFddaCarry):
        return
    if not isinstance(group, SpectralFddaCarry) or any(
            getattr(group, f.name, None) is None for f in fields(SpectralFddaCarry)):
        raise ValueError("restart spectral FDDA is missing analysis/held tendency leaves (E78)")


def validate_grid_fdda(group, namelist):
    """Grid nudging restarts need the analysis cursor, rates and held heat flux."""
    from dataclasses import fields
    from gpuwrf.assimilation.fdda_grid import GridFddaInputs, GridFddaCarry, GridFddaHistoryCarry
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs, SpectralFddaCarry
    import numpy as np

    inputs = getattr(namelist, "data_assimilation", None)
    if isinstance(inputs, SpectralFddaInputs):
        return
    selected = isinstance(inputs, GridFddaInputs)
    if isinstance(namelist, Mapping):
        raw = namelist.get("fdda", {}).get("grid_fdda", 0)
        selected = any(int(code) == 1 for code in (raw if isinstance(raw, (list, tuple)) else (raw,)))
    if isinstance(group, SpectralFddaCarry) and not selected:
        return
    if not selected and not isinstance(group, GridFddaCarry):
        return
    if not isinstance(group, GridFddaHistoryCarry) or any(getattr(group, f.name, None) is None for f in fields(GridFddaHistoryCarry)):
        raise ValueError("restart grid FDDA is missing analysis/held HFX_FDDA or tendency leaves (E78)")
    shape = np.shape(group.t_old)
    grid = getattr(namelist, "grid", None)
    if len(shape) != 3 or (grid is not None and shape != (grid.nz, grid.ny, grid.nx)):
        raise ValueError("restart grid FDDA domain shape differs (E78)")
    nz, ny, nx = shape
    dtype = np.asarray(group.t_old).dtype
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError("restart grid FDDA REAL dtype differs (E78)")
    for f in fields(GridFddaHistoryCarry):
        value = np.asarray(getattr(group, f.name))
        expected = ((nz, ny, nx + 1) if f.name.startswith('u_') or f.name == 'rundgdten' else
                    (nz, ny + 1, nx) if f.name.startswith('v_') or f.name == 'rvndgdten' else shape)
        if value.shape != expected or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"restart grid FDDA {f.name} shape/dtype/nonfinite differs (E78)")

def validate_observations(group, namelist):
    """Require the complete rolling frame and bind it to the input stream."""
    inputs = getattr(namelist, 'observation_nudging', None)
    if isinstance(namelist, Mapping):
        inputs = namelist.get('observation_nudging')
        selected = any(int(x) for x in (namelist.get('fdda', {}).get('obs_nudge_opt', 0),)
                       if not isinstance(x, (list,tuple)))
        raw = namelist.get('fdda', {}).get('obs_nudge_opt', 0)
        if isinstance(raw,(list,tuple)):selected = any(int(x) for x in raw)
    else:
        selected = inputs is not None
    if group is None:
        if selected or inputs is not None:
            raise ValueError('restart is missing held observation VAROBS/RKO/SAVWT (E78)')
        return
    from dataclasses import fields
    from gpuwrf.assimilation.obs_driver import ObservationCarry
    import numpy as np
    if not isinstance(group,ObservationCarry) or any(getattr(group,f.name,None) is None for f in fields(ObservationCarry)):
        raise ValueError('restart observation carry schema is incomplete (E78)')
    if any(not np.isfinite(np.asarray(getattr(group,f.name))).all() for f in fields(ObservationCarry)):
        raise ValueError('restart observation carry contains NaN/Inf')
    if inputs is not None:
        expected=np.asarray(inputs.fingerprint,np.uint8)
        if not np.array_equal(np.asarray(group.fingerprint),expected):
            raise ValueError('restart observation input/options fingerprint changed')
        if group.varobs.shape!=(inputs.max_reports,inputs.rko.shape[1],6) or group.rko.shape!=group.varobs.shape[:2]:
            raise ValueError('restart observation rolling-frame shape changed')

def _urban_real(value,name,shape=None):
    """Host checkpoint check, with bounded scratch instead of a grid-size mask."""
    import numpy as np
    if value is None:
        raise ValueError(f'restart BEP/BEM missing {name} (E78)')
    array=np.asarray(value)
    if array.dtype!=np.float32 or (shape is not None and array.shape!=shape):
        raise ValueError(f'restart BEP/BEM {name} shape/dtype differs (E78)')
    # flat slices bound the temporary to 4096 REALs plus a finite mask; no
    # device allocation, timestep check, or full-grid boolean temporary.
    if any(not np.isfinite(array.flat[start:start+4096]).all()
           for start in range(0,array.size,4096)):
        raise ValueError(f'restart BEP/BEM nonfinite {name} (E78)')
    return array


def _urban_context(namelist,grid_shape=None,nz=None):
    get=(namelist.get if isinstance(namelist,Mapping)
         else lambda key,default=None:getattr(namelist,key,default))
    bundle=get('urban_bundle');grid=get('grid')
    if grid_shape is None:
        if bundle is not None:grid_shape=tuple(bundle.shape)
        elif grid is not None:grid_shape=(int(grid.ny),int(grid.nx))
    if nz is None and grid is not None:nz=int(grid.nz)
    return bundle,grid_shape,nz


def validate_urban_bep(group,namelist,*,grid_shape=None,nz=None):
    """Never resume BEP/BEM by silently reinitializing building temperatures."""
    option=physics_option(namelist,'sf_urban_physics')
    if option not in (2,3) and group is None:return
    from gpuwrf.coupling.urban_multilayer_hook import UrbanMultilayerCarry
    from gpuwrf.physics.urban_bep_model import BepState
    from gpuwrf.physics.urban_bem_model import BemState
    if not isinstance(group,UrbanMultilayerCarry):
        raise ValueError('restart BEP/BEM is missing complete urban_bep carry (E78)')
    expected=BemState if option==3 else BepState
    if not isinstance(group.state,expected):
        raise ValueError('restart BEP/BEM material/indoor state differs from selected scheme (E78)')
    from gpuwrf.physics.urban_bep_blend import UrbanPblSources
    if not isinstance(group.sources,UrbanPblSources):
        raise ValueError('restart BEP/BEM missing urban_bep sources (E78)')
    from gpuwrf.coupling.urban_surface_layer import UrbanSurfaceLayer
    if not isinstance(group.surface_layer,UrbanSurfaceLayer) or any(x is None for x in group.surface_layer):
        raise ValueError('restart BEP/BEM is missing selected surface-layer diagnostics (E78)')
    import numpy as np
    bundle,grid_shape,nz=_urban_context(namelist,grid_shape,nz)
    green_roof=getattr(group,'green_roof',None)
    if (green_roof is None)!=(option!=3):
        raise ValueError('restart BEP+BEM QGR_URB3D presence differs from the selected scheme (E78)')
    if green_roof is not None:
        green_roof=_urban_real(green_roof,'green_roof')
        if grid_shape is not None and green_roof.shape!=tuple(grid_shape):
            raise ValueError('restart BEP+BEM QGR_URB3D grid shape differs (E78)')
    fraction=_urban_real(group.fraction,'fraction')
    if fraction.ndim!=1:
        raise ValueError('restart BEP/BEM fraction shape differs (E78)')
    count=len(fraction)
    if bundle is not None and (count!=len(bundle.indices) or option!=bundle.option):
        raise ValueError('restart BEP/BEM urban partition/option differs (E78)')
    # Validate required fields before flattening: pytree leaves omit None.
    for name in group.state._fields:
        if getattr(group.state,name) is None:
            raise ValueError(f'restart BEP/BEM missing state.{name} (E78)')
    roof=_urban_real(group.state.roof,'state.roof')
    if roof.ndim!=4 or roof.shape[2]<1:
        raise ValueError('restart BEP/BEM roof urban-level axis invalid (E78)')
    levels=roof.shape[2]
    if bundle is not None and levels!=bundle.geometry.n:
        raise ValueError('restart BEP/BEM urban levels differ from geometry (E78)')
    shapes=dict(roof=(count,2,levels,10),ground=(count,2,10),
        roof_heat=(count,2,levels),ground_heat=(count,2))
    if option==2:
        shapes.update(wall=(count,4,levels,10),wall_heat=(count,4,levels))
    else:
        room=_urban_real(group.state.temperature,'state.temperature')
        if room.ndim!=3 or room.shape[1]<1 or room.shape[2]<1:
            raise ValueError('restart BEP/BEM building/floor axes invalid (E78)')
        buildings,floors=room.shape[1:]
        if bundle is not None and (buildings!=len(bundle.bem.levels) or floors!=max(bundle.bem.levels)):
            raise ValueError('restart BEP/BEM building/floor geometry differs (E78)')
        shapes.update(wall=(count,buildings,4,levels,10),wall_heat=(count,buildings,4,levels),
            temperature=(count,buildings,floors),humidity=(count,buildings,floors),
            window=(count,buildings,4,levels),window_heat=(count,buildings,4,levels),
            floor=(count,buildings,2,floors-1,10),under_ground=(count,buildings,2,10),
            roof_water=(count,2,levels),ground_water=(count,2),
            roof_latent=(count,2,levels),ground_latent=(count,2),pv_temperature=(count,2,levels))
        if not isinstance(group.state.extra,Mapping):
            raise ValueError('restart BEP/BEM disabled extra state mapping missing (E78)')
        for name,value in group.state.extra.items():
            extra=_urban_real(value,'state.extra.'+str(name))
            if extra.ndim<1 or extra.shape[0]!=count:
                raise ValueError('restart BEP/BEM extra state column axis differs (E78)')
    for name,shape in shapes.items():_urban_real(getattr(group.state,name),'state.'+name,shape)
    first=_urban_real(group.sources.a_u,'sources.a_u')
    if first.ndim!=2 or first.shape[1]<1:
        raise ValueError('restart BEP/BEM source level axis invalid (E78)')
    if nz is not None and first.shape[1]!=nz:
        raise ValueError('restart BEP/BEM source levels differ from domain (E78)')
    nz=first.shape[1]
    for name in group.sources._fields:
        shape=(count,) if name=='fraction' else (count,nz+1 if name=='surface' else nz)
        _urban_real(getattr(group.sources,name),'sources.'+name,shape)
    if not np.array_equal(np.asarray(group.sources.fraction),fraction):
        raise ValueError('restart BEP/BEM held source fraction differs (E78)')
    surface=_urban_real(group.surface_layer.br,'surface-layer diagnostics.br')
    if grid_shape is None:grid_shape=surface.shape
    if len(grid_shape)!=2 or min(grid_shape)<1:
        raise ValueError('restart BEP/BEM surface grid shape invalid (E78)')
    for name in group.surface_layer._fields:
        _urban_real(getattr(group.surface_layer,name),'surface-layer diagnostics.'+name,tuple(grid_shape))


def validate_urban_radiation(rad,namelist,*,grid_shape=None):
    if physics_option(namelist,'sf_urban_physics') not in (2,3):return
    held=getattr(rad,'urban_radiation',None)
    if not isinstance(held,(tuple,list)) or len(held)!=4 or any(x is None for x in held):
        raise ValueError('restart BEP/BEM lacks held direct/diffuse radiation and solar angles (E78)')
    _,grid_shape,_=_urban_context(namelist,grid_shape)
    direct=_urban_real(held[0],'direct/diffuse radiation.direct')
    if grid_shape is None:grid_shape=direct.shape
    if len(grid_shape)!=2 or min(grid_shape)<1:
        raise ValueError('restart BEP/BEM radiation grid shape invalid (E78)')
    for index,name in enumerate(('direct','diffuse','declination','hour_angle')):
        value=_urban_real(held[index],'direct/diffuse radiation.'+name)
        allowed=(tuple(grid_shape),) if index<2 else ((),tuple(grid_shape))
        if value.shape not in allowed:
            raise ValueError(f'restart BEP/BEM radiation.{name} shape differs (E78)')

def ssib_active(namelist):
    """SSiB needs an explicit selector; None keeps the inferred legacy LSM."""
    options = namelist.get('physics', namelist) if isinstance(namelist, Mapping) else namelist
    value = options.get('sf_surface_physics') if isinstance(options, Mapping) else getattr(options, 'sf_surface_physics', None)
    return value is not None and int(value) == 8


def validate_ssib(group, radiation, namelist, schema_version, shape, *, active=None, driver_fields=None):
    """SSiB complete snow/soil + held radiation E78 admission on all restart APIs."""
    if active is None:
        active=ssib_active(namelist)
    if not active and group is None and radiation is None:
        return
    if not active:
        raise ValueError('SSiB restart carry/selection mismatch (E78)')
    from gpuwrf.physics.lsm_ssib import validate_ssib_restart
    import numpy as np
    validate_ssib_restart(group,schema_version,shape,active=True)
    if not isinstance(radiation,tuple) or len(radiation)!=3:
        raise ValueError('SSiB restart missing held radiation (E78)')
    for value in radiation:
        if np.shape(value)!=tuple(shape) or np.asarray(value).dtype!=np.float32 or not np.all(np.isfinite(value)):
            raise ValueError('SSiB restart held radiation shape/dtype/nonfinite (E78)')

    if driver_fields is not None:
        wind=driver_fields.get('wspd')
        if np.shape(wind)!=tuple(shape) or np.asarray(wind).dtype!=np.float32 or not np.isfinite(wind).all():
            raise ValueError('SSiB restart missing/invalid held WSPD (E78)')
        from gpuwrf.runtime.operational_state import HistoryDiagnostics
        from gpuwrf.runtime.noahmp_precipitation import NoahMPPrecipitation
        for name,kind in (('history',HistoryDiagnostics),('precipitation',NoahMPPrecipitation)):
            value=driver_fields.get(name)
            if not isinstance(value,kind):raise ValueError(f'SSiB restart missing {name} carry (E78)')
            for field in value._fields:
                leaf=getattr(value,field)
                if name=='history' and leaf is None and not history_field_required(field,namelist):continue
                if name=='precipitation' and field=='sr' and leaf is None:continue  # Noah-classic-only SR channel
                if np.shape(leaf)!=tuple(shape) or np.asarray(leaf).dtype!=np.float32 or not np.isfinite(leaf).all():
                    raise ValueError(f'SSiB restart invalid {name}.{field} (E78)')
