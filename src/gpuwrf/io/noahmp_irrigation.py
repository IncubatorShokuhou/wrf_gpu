"""Initialize Noah-MP irrigation from WRF maps/tables outside the timestep."""
from dataclasses import replace
import re

import jax.numpy as jnp
import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.physics.noahmp.irrigation import IrrigationParameters, IrrigationStatic, seed_irrigation


def bind_irrigation(namelist, run_dir, domain, *, state):
    options = namelist.noahmp_options
    if options is None or options.opt_irr == 0:
        return namelist
    if not namelist.use_noahmp or namelist.noahmp_energy_params is None:
        raise ValueError("Noah-MP irrigation requires prognostic Noah-MP and gathered soil parameters")
    if namelist.sf_lake_physics:
        raise ValueError("Noah-MP irrigation plus compact lake tiles is not qualified")
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.physics.noahmp.water_hydro import _soil_param
    shape = tuple(state.t_skin.shape)
    ep, static = namelist.noahmp_energy_params, namelist.noahmp_static
    with Dataset(Gen2Run(run_dir).wrfinput_file(domain), "r") as ds:
        def field(name, *, required=False, dtype=np.float32):
            candidates = (name.upper(),name,name.lower())  # canonical WRF names win over aliases
            key = next((key for key in candidates if key in ds.variables),None)
            if key is None:
                if required:
                    raise ValueError(f"Noah-MP irrigation requires wrfinput field {name}")
                return jnp.zeros(shape,jnp.int32 if dtype==np.int32 else jnp.float32)
            raw = np.ma.asarray(ds[key][:])
            if np.ma.getmaskarray(raw).any() or not np.isfinite(np.asarray(raw)).all():
                raise ValueError(f"Noah-MP irrigation field {name} contains missing/nonfinite data")
            if dtype==np.int32 and np.any(np.asarray(raw)!=np.floor(np.asarray(raw))):
                raise ValueError(f"Noah-MP irrigation counter {name} must be integer-valued")
            arr = np.asarray(raw,dtype=dtype).squeeze()
            if np.ma.getmaskarray(raw).any() or arr.shape!=shape or not np.isfinite(arr).all():
                raise ValueError(f"Noah-MP irrigation field {name} must be finite with surface shape {shape}")
            return jnp.asarray(arr)
        irfract = field("IRFRACT",required=True)
        fractions = [field(key) for key in ("SIFRACT","MIFRACT","FIFRACT")]
        for values in (irfract,*fractions):
            if np.any(np.asarray(values)<0.) or np.any(np.asarray(values)>1.):
                raise ValueError("WRF irrigation map fractions must be between zero and one")
        planting = field("planting",required=options.opt_irr==2)
        harvest = field("harvest",required=options.opt_irr==2)
        dataset = str(ds.getncattr("MMINLU")).strip()
        ivg = jnp.asarray(static.ivgtyp,jnp.int32)
        if dataset == "USGS":
            crop = (ivg>=3)&(ivg<=6)
        elif dataset == "MODIFIED_IGBP_MODIS_NOAH":
            crop = (ivg==12)|(ivg==14)
        else:
            raise ValueError(f"WRF irrigation cropland mapping is not defined for MMINLU={dataset!r}")
        initial=seed_irrigation(shape)
        names=("irnumsi","irnummi","irnumfi","irsivol","irmivol","irfivol","ireloss","irrsplh")
        initial=initial._replace(**{name:field(name,dtype=np.int32 if name.startswith("irnum") else np.float32)
                                    for name in names})
        # NOAHMP_INIT:2262-2276 resets one method (the first ELSE-IF wins for 0).
        resets=("irnumsi","irwatsi","ireloss","irrsplh") if options.opt_irrm in (0,1) else (
               ("irnummi","irwatmi","irmivol") if options.opt_irrm==2 else ("irnumfi","irwatfi","irfivol"))
        zero=seed_irrigation(shape)
        initial=initial._replace(**{name:getattr(zero,name) for name in resets})
    # Only this opt-in table block is read; default tables/treedef stay exact.
    text = "\n".join(line.split("!",1)[0] for line in (wrf_run_dir()/"MPTABLE.TBL").read_text().splitlines())
    block = re.search(r"&noahmp_irrigation_parameters\b(.*?)/",text,re.I|re.S)
    if block is None:
        raise ValueError("MPTABLE.TBL lacks &noahmp_irrigation_parameters")
    scalars = {}
    for name in IrrigationParameters._fields[:9]:
        match = re.search(rf"\b{name}\s*=\s*([-+\d.eEdD]+)",block[1],re.I)
        if match is None:
            raise ValueError(f"MPTABLE.TBL lacks irrigation parameter {name}")
        value = float(match[1].lower().replace("d","e"))
        if not np.isfinite(value):
            raise ValueError(f"MPTABLE.TBL irrigation parameter {name} is nonfinite")
        scalars[name] = jnp.float32(value)
    if not 0.<=float(scalars["filoss"])<1.:
        raise ValueError("Noah-MP FILOSS must be in [0,1)")
    soil = jnp.zeros((4,*shape),jnp.float32)
    params = IrrigationParameters(**scalars,pltday=planting,hsday=harvest,
        nroot=jnp.asarray(ep.nroot_cell if ep.nroot_cell is not None else jnp.full(shape,ep.nroot),jnp.int32),
        smcwlt=jnp.asarray(ep.smcwlt,jnp.float32),smcref=jnp.asarray(ep.smcref,jnp.float32),
        smcmax=jnp.asarray(ep.smcmax,jnp.float32),bexp=jnp.asarray(ep.bexp,jnp.float32),
        dksat=jnp.asarray(_soil_param(static.parameters,"dksat",static,soil,1.e-6),jnp.float32),
        dwsat=jnp.asarray(_soil_param(static.parameters,"dwsat",static,soil,1.e-5),jnp.float32))
    return replace(namelist,noahmp_irrigation_static=IrrigationStatic(params,irfract,*fractions,crop,initial))


def validate_restart(group, config, version, shape, *, requested=None):
    from gpuwrf.physics.noahmp.irrigation import IrrigationState
    if not isinstance(config,(list,tuple)) or len(config)!=2 or config[0] not in (0,1,2,3) or config[1] not in (0,1,2,3):
        raise ValueError("Noah-MP irrigation restart configuration is malformed (E78)")
    if requested is not None and tuple(config)!=tuple(requested):
        raise ValueError("Noah-MP irrigation restart configuration differs (E78)")
    enabled=bool(config[0])
    if not enabled and group is None:
        return
    if not enabled or version!=1 or not isinstance(group,IrrigationState):
        raise ValueError("Noah-MP irrigation restart schema/selection is missing or inconsistent (E78)")
    for index,value in enumerate(group):
        arr=np.asarray(value)
        dtype=np.int32 if index in (0,1,2,11) else np.float32
        if arr.shape!=tuple(shape) or arr.dtype!=dtype or not np.isfinite(arr).all():
            raise ValueError("Noah-MP irrigation restart needs all REAL/counter leaves with the surface shape (E78)")
