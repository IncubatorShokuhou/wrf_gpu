"""Noah-MP MMF selection and real-geography initialization.

Selection checks are backend-dark. Model imports occur only in the loader,
after the CLI has configured its allocator/backend.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

SCHEMA_VERSION = 1


def configuration(namelist):
    """Small host selection packet, also authenticated by nested restart."""
    from collections.abc import Mapping
    if isinstance(namelist, Mapping):
        active = bool(namelist.get('noahmp_mmf_enabled', selected(namelist)))
        inf = namelist.get('noahmp_mmf_opt_inf', (namelist.get('noah_mp') or {}).get('opt_inf', 1))
        dt = namelist.get('dt_s')
    else:
        active = bool(getattr(namelist, 'noahmp_mmf_enabled', False))
        inf = getattr(namelist, 'noahmp_mmf_opt_inf', 1)
        dt = getattr(namelist, 'dt_s', None)
    if isinstance(inf, (list, tuple)):
        inf = inf[0]
    config = dict(noahmp_mmf_enabled=active, noahmp_mmf_opt_inf=int(inf))
    if active and dt is not None:
        config['dt_s'] = float(dt)
    return config


def validate_restart(bundle, namelist=None, version=None, shape=None):
    """Validate host leaves before restoration can narrow their precision (E78)."""
    import numpy as np
    expected = configuration(namelist) if namelist is not None else None
    active = bundle is not None
    if expected is not None and active != expected['noahmp_mmf_enabled']:
        raise ValueError('MMF restart selection/carry differs (E78)')
    if not active:
        if version is not None:
            raise ValueError('MMF restart marker has no complete carry (E78)')
        return
    from gpuwrf.physics.noahmp.groundwater import MMFBundle, GroundwaterStatic, GroundwaterCarry
    if version != SCHEMA_VERSION or not isinstance(bundle, MMFBundle):
        raise ValueError('MMF restart schema differs (E78)')
    if not isinstance(bundle.static, GroundwaterStatic) or not isinstance(bundle.carry, GroundwaterCarry):
        raise ValueError('MMF restart geometry/state schema differs (E78)')
    if bundle.opt_inf not in (1, 2) or (expected is not None and
            bundle.opt_inf != expected['noahmp_mmf_opt_inf']):
        raise ValueError('MMF restart infiltration selection differs (E78)')
    if type(bundle.acrech) is not int or type(bundle.acqspring) is not int or bundle.acrech != 0 or bundle.acqspring != 0:
        raise ValueError('MMF restart ACRECH/ACQSPRING requires inactive NOAHMP_ACC_DT=0 (E78)')
    shape = tuple(shape) if shape is not None else np.asarray(bundle.carry.zwt).shape
    if len(shape) != 2:
        raise ValueError('MMF restart requires a mass-grid shape (E78)')
    for group in (bundle.static, bundle.carry):
        for name, value in zip(group._fields, group, strict=True):
            arr = np.asarray(value)
            dims = (4, *shape) if name == 'smoiseq' else shape
            if arr.shape != dims or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError(f'MMF restart {name} must be finite REAL4 with shape {dims} (E78)')
    mask = np.asarray(bundle.eligibility)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError('MMF restart eligibility differs (E78)')
    for name in ('fsatmx', 'seconds'):
        arr = np.asarray(getattr(bundle, name))
        if arr.shape != () or arr.dtype != np.float32 or not np.isfinite(arr):
            raise ValueError(f'MMF restart {name} must be a finite REAL4 scalar (E78)')
    if not 0 <= float(bundle.fsatmx) <= 1 or float(bundle.seconds) <= 0:
        raise ValueError('MMF restart physical controls differ (E78)')
    step = np.asarray(bundle.stepwtd)
    if step.shape != () or step.dtype != np.int32 or int(step) < 1:
        raise ValueError('MMF restart STEPWTD must be a positive INTEGER4 scalar (E78)')
    if expected is not None and 'dt_s' in expected:
        dt = np.float32(expected['dt_s'])
        if not np.isfinite(dt) or dt <= 0 or int(step) != max(int(np.floor(
                np.float32(np.float32(bundle.seconds) / dt) + np.float32(.5))), 1):
            raise ValueError('MMF restart cadence differs from domain DT (E78)')
    if np.any(np.asarray(bundle.static.area) <= 0):
        raise ValueError('MMF restart requires positive cell areas (E78)')


def history_fields(bundle, land):
    """Held Registry fields; no physics recomputation or host readback."""
    fields = {name.upper(): value for name, value in zip(bundle.static._fields, bundle.static, strict=True)
              if name != 'topo'}
    fields.update({name.upper(): value for name, value in zip(bundle.carry._fields, bundle.carry, strict=True)})
    fields['SMCWTD'] = land.smcwtd
    # NOAHMP_INIT sets WA/WT=0 for OPT_RUN=5. NOAHMP_ACC_DT>0 is refused.
    zero = land.smcwtd * 0
    fields.update(WA=zero, WT=zero, ACRECH=zero, ACQSPRING=zero)
    return fields


def validate_land(bundle, land, shape):
    """MMF's soil/deep moisture lives in the companion Noah-MP land group."""
    if bundle is None:
        return
    from collections.abc import Mapping
    import numpy as np
    for name in ('smois','sh2o','smcwtd'):
        value = land.get(name) if isinstance(land, Mapping) else getattr(land, name, None)
        arr = np.asarray(value)
        dims = tuple(shape) if name=='smcwtd' else (4,*shape)
        if arr.shape != dims or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f'MMF restart requires complete REAL4 Noah-MP {name} with shape {dims} (E78)')


def selected(namelist) -> bool:
    value = (namelist.get('noah_mp') or {}).get('opt_run', 3)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 3
    return value == 5


def refusals(namelist, domains) -> list[str]:
    if not selected(namelist):
        return []
    from gpuwrf.io.namelist_binding import namelist_value
    import numpy as np
    errors = []
    if len(domains) != 1:
        errors.append('opt_run=5 nested groundwater interpolation/feedback is not qualified; use one domain')
    acc = (namelist.get('noah_mp') or {}).get('noahmp_acc_dt', 0)
    if isinstance(acc, (list, tuple)):
        acc = acc[0]
    if acc != 0:
        errors.append('opt_run=5 currently requires noahmp_acc_dt=0; active accumulated recharge is not qualified')
    for domain in domains:
        for key, allowed in [('sf_surface_physics', (4,)), ('num_soil_layers', (None, 4)),
                             ('sf_lake_physics', (None, 0)), ('sf_urban_physics', (None, 0))]:
            value = namelist_value(namelist, key, domain)
            if value not in allowed:
                errors.append(f'opt_run=5 ({domain}) requires {key} in {allowed}; got {value}')
        wtddt = namelist_value(namelist, 'wtddt', domain)
        wtddt = 30. if wtddt is None else float(wtddt)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            seconds = np.float32(wtddt) * np.float32(60.)
        if not math.isfinite(wtddt) or wtddt <= 0 or not np.isfinite(seconds) or seconds <= 0:
            errors.append(f'opt_run=5 ({domain}) requires finite positive wtddt (minutes)')
    inf = (namelist.get('noah_mp') or {}).get('opt_inf', 1)
    if isinstance(inf, (list, tuple)):
        inf = inf[0]
    if inf not in (1, 2):
        errors.append('opt_run=5 requires opt_inf=1 or 2')
    return errors


def preflight(input_dir, namelist, domains):
    """Backend-dark real-file gate shared by public CLI and model loader."""
    if not selected(namelist):
        return
    import numpy as np
    from gpuwrf.io.netcdf_lock import Dataset
    errors = refusals(namelist, domains)
    if errors:
        raise ValueError('; '.join(errors))
    for domain in domains:
        with Dataset(Path(input_dir) / ('wrfinput_' + domain)) as ds:
            required = ('FDEPTH', 'EQZWT', 'RECHCLIM', 'RIVERBED', 'HGT', 'MAPFAC_MX', 'MAPFAC_MY',
                        'XLAND', 'ISLTYP', 'IVGTYP')
            missing = set(required) - set(ds.variables)
            if missing:
                raise ValueError(f'MMF missing {sorted(missing)}; use GEOGRID.TBL.ARW.noahmp and NCAR groundwater data')
            values = {n: np.asarray(ds[n][0]) for n in required}
            shape = values['HGT'].shape
            if len(shape) != 2 or min(shape) < 3 or not hasattr(ds, 'ISICE'):
                raise ValueError('MMF requires at least a 3x3 mass grid and original ISICE metadata')
            for name, value in values.items():
                real = np.asarray(value, np.float32)
                if value.shape != shape or not np.isfinite(value).all() or not np.isfinite(real).all():
                    raise ValueError(f'MMF {name} must be finite REAL4 on the mass grid')
            if 'SEAICE' in ds.variables and np.any(np.asarray(ds['SEAICE'][0]) != 0):
                raise ValueError('MMF requires ice-free geography')
            if np.any((values['XLAND'] < 1.5) & (values['IVGTYP'] == int(ds.ISICE))):
                raise ValueError('MMF glacier coupling is not qualified; use an ice-free domain')
            eligible = (values['XLAND'] < 1.5) & (values['IVGTYP'] != int(ds.ISICE))
            if not np.any((values['FDEPTH'] > 0) & eligible):
                raise ValueError('MMF geography inactive: no positive land FDEPTH; use NCAR groundwater data')
            if (np.any(values['ISLTYP'] < 1) or np.any(values['ISLTYP'] > 19)
                    or np.any(values['ISLTYP'] != np.floor(values['ISLTYP']))):
                raise ValueError('MMF requires integer STAS soil categories 1..19')
            if np.any(values['MAPFAC_MX'] <= 0) or np.any(values['MAPFAC_MY'] <= 0):
                raise ValueError('MMF requires positive map factors')


def _fsatmx(table_path: Path) -> float:
    text = table_path.read_text()
    block = re.search(r'&noahmp_global_parameters\b(.*?)(?:\n\s*/)', text, re.I | re.S)
    if block is None:
        raise ValueError('MMF requires noahmp_global_parameters in MPTABLE.TBL')
    match = re.search(r'\bFSATMX\s*=\s*([-+\d.eEdD]+)', block.group(1), re.I)
    if match is None:
        raise ValueError('MMF requires actual MPTABLE FSATMX')
    value = float(match.group(1).replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('MMF MPTABLE FSATMX must be finite and in [0,1]')
    return value


def load_mmf(run, domain, land, static, dt_s, *, table_dir=None):
    """Return initialized (land, MMFBundle), with original REAL4/NINT cadence."""
    from gpuwrf.io.namelist_binding import namelist_value
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.physics.noahmp.groundwater import MMFBundle, initialize
    import jax.numpy as jnp
    import numpy as np

    errors = refusals(run.namelist, (domain,))
    if errors:
        raise ValueError('; '.join(errors))
    wtddt = namelist_value(run.namelist, 'wtddt', domain)
    wtddt = np.float32(30. if wtddt is None else wtddt)
    seconds = wtddt * np.float32(60.)
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('MMF requires a finite positive domain timestep')
    ratio = np.float32(seconds / dt)
    if not np.isfinite(ratio) or ratio >= np.iinfo(np.int32).max:
        raise ValueError('MMF WTDDT/DT cadence must fit WRF INTEGER4')
    stepwtd = max(int(np.floor(ratio + np.float32(.5))), 1)
    with run.input_read_scope(domain):
        present = set(run.wrfinput_variables(domain))
        required = ('FDEPTH', 'EQZWT', 'RECHCLIM', 'RIVERBED', 'HGT', 'MAPFAC_MX', 'MAPFAC_MY')
        missing = set(required) - present
        if missing:
            raise ValueError(f'MMF input fields missing: {sorted(missing)}; '
                             'prepare WPS with GEOGRID.TBL.ARW.noahmp and NCAR groundwater data')
        values = {name: np.squeeze(np.asarray(run.load_wrfinput(domain, name, lazy=False), np.float32))
                  for name in required}
        seaice = (np.squeeze(np.asarray(run.load_wrfinput(domain, 'SEAICE', lazy=False), np.float32))
                  if 'SEAICE' in present else np.zeros_like(values['HGT']))
        grid = run.grid(domain)
    shape = tuple(land.smcwtd.shape)
    if any(value.shape != shape or not np.isfinite(value).all() for value in values.values()):
        raise ValueError('MMF geography must be finite and match the domain mass grid')
    if not np.isfinite(seaice).all() or np.any(seaice > 0):
        raise ValueError('MMF requires the qualified ice-free Noah-MP coupling')
    if np.any((np.asarray(static.xland) < 1.5) &
              (np.asarray(static.ivgtyp) == int(static.parameters.isice))):
        raise ValueError('MMF glacier coupling is not qualified; use an ice-free domain')
    eligible = ((np.asarray(static.xland) < 1.5) & (seaice < .5)
                & (np.asarray(static.ivgtyp) != int(static.parameters.isice)))
    if not np.any((values['FDEPTH'] > 0) & eligible):
        raise ValueError('MMF groundwater geography is absent/inactive (no positive land FDEPTH); '
                         'rerun WPS with GEOGRID.TBL.ARW.noahmp and NCAR groundwater data')
    soil = np.asarray(static.isltyp)
    if np.any((soil < 1) | (soil > 19)):
        raise ValueError('MMF requires WRF STAS soil categories 1..19')
    mx, my = values['MAPFAC_MX'], values['MAPFAC_MY']
    if np.any(mx <= 0) or np.any(my <= 0):
        raise ValueError('MMF requires positive map factors')
    # Both map factors matter (non-Lambert cases); WRF NOAHMP_INIT:2151.
    dx, dy = np.float32(grid.dx_m), np.float32(grid.dy_m)
    area = (dx * dy) / (mx * my)
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('MMF requires finite positive REAL4 cell areas')
    result = initialize(jnp.asarray(values['FDEPTH'], jnp.float32),
        jnp.asarray(area, jnp.float32), jnp.asarray(values['HGT'], jnp.float32),
        jnp.asarray(values['EQZWT'], jnp.float32), jnp.asarray(values['RECHCLIM'], jnp.float32),
        jnp.asarray(values['RIVERBED'], jnp.float32), jnp.asarray(land.smois, jnp.float32),
        jnp.asarray(land.sh2o, jnp.float32), jnp.asarray(static.dzs, jnp.float32),
        static.isltyp, static.ivgtyp, static.parameters, jnp.asarray(seconds, jnp.float32))
    geometry, carry, smc, sh2o, aux = result
    tables = Path(table_dir) if table_dir is not None else wrf_run_dir()
    bundle = MMFBundle(geometry, carry, jnp.asarray(eligible),
                       jnp.asarray(_fsatmx(tables / 'MPTABLE.TBL'), jnp.float32),
                       jnp.asarray(seconds, jnp.float32), jnp.asarray(stepwtd, jnp.int32),
                       opt_inf=int(namelist_value(run.namelist, 'opt_inf', domain) or 1))
    return land.replace(smois=smc, sh2o=sh2o, smcwtd=aux), bundle
