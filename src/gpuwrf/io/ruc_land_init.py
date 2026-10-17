"""RUC model init from real.exe-prepared wrfinput (no second soil interpolation).

WRF share/module_soil_pre.F process_soil_real interpolates meteorological soil
inputs in real.exe. phys/module_physics_init.F lsminit then calls RUCLSMINIT on
those existing RUC profiles. Noah 4-layer input is not a RUC initial condition.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset


RUC_SOIL_LEVELS = {
    6: (0., .05, .20, .40, 1.60, 3.),
    9: (0., .01, .04, .10, .30, .60, 1., 1.60, 3.),
}


def load_ruc_land(run_dir, domain, *, num_soil_layers, dt, p1, table_dir=None):
    """Return RucStaticBundle/RucLandState; refuse incompatible inputs at init.

    The requested soil count is the scalar WRF &physics num_soil_layers. ZS and
    profile dimensions must match the grid prepared by real.exe with sf=3.
    Tables use the normal configurable pristine WRF run-directory resolver.
    """
    from gpuwrf.coupling.ruc_surface_hook import build_ruc_bundles, ruc_unsupported_mask

    nzs = int(num_soil_layers)
    if nzs not in RUC_SOIL_LEVELS:
        raise ValueError(f"RUC requires num_soil_layers=6 or 9; got {nzs}")
    path = Path(run_dir) / f"wrfinput_{domain}"
    required = ("ZS", "IVGTYP", "ISLTYP", "XLAND", "TMN", "SHDMIN", "SHDMAX",
                "ALBBCK", "VEGFRA", "TSK", "TSLB", "SMOIS", "SNOALB")
    optional = ("SEAICE", "XICE", "SNOW", "SNOWH", "SNOWC", "CANWAT", "LAI", "LAKEMASK",
                "LU_INDEX", "SOILT1", "QVG", "QCG", "QSFC")
    with Dataset(path, "r") as ds:
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"{path}: RUC wrfinput missing {missing}; prepare inputs with real.exe sf=3")
        fields = {}
        for name in required + optional:
            if name in ds.variables:
                var = ds.variables[name]
                arr = np.asarray(var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:])
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{path}: nonfinite RUC input {name}")
                fields[name] = arr
        attrs = {name: getattr(ds, name, None) for name in ("MMINLU", "ISWATER", "ISICE")}
        if any(value is None for value in attrs.values()):
            raise ValueError(f"{path}: RUC requires MMINLU/ISWATER/ISICE metadata")
        source_scheme = getattr(ds, "SF_SURFACE_PHYSICS", 3)
        if int(source_scheme) != 3:
            raise ValueError(f"{path}: wrfinput SF_SURFACE_PHYSICS={source_scheme}; prepare RUC inputs with real.exe sf=3")
        # Generic LANDUSE_INIT runs before lsminit in WRF. Reuse its existing
        # host-only seasonal table implementation; the CLI refuses usemonalb.
        from gpuwrf.io.land_history import _landuse_init
        from gpuwrf.config.paths import wrf_run_dir
        # LANDUSE_INIT uses NINT(LU_INDEX), with the zero/no-data class mapped
        # to the water class before looking up the table.
        lu = np.floor(fields.get("LU_INDEX", fields["IVGTYP"]) + np.float32(.5)).astype(np.int32)
        lu = np.where(lu == 0, int(attrs["ISWATER"]), lu)
        albbck, albedo, emiss = _landuse_init(
            ds, Path(table_dir) if table_dir is not None else wrf_run_dir(),
            lu, fields.get("SNOWC", 0.),
        )
        fields["ALBBCK"] = albbck
        fields["XLAND"] = np.where(lu == int(attrs["ISWATER"]), 2., 1.).astype(np.float32)
    zs = np.asarray(fields.pop("ZS")).reshape(-1)
    expected = np.asarray(RUC_SOIL_LEVELS[nzs], np.float32)
    if zs.shape != expected.shape or not np.allclose(zs, expected, rtol=0., atol=1.e-7):
        raise ValueError(f"{path}: RUC ZS must match num_soil_layers={nzs} levels {expected.tolist()}; got {zs.tolist()}")
    shape = np.shape(fields["TSK"])
    if len(shape) != 2 or np.shape(p1) != shape:
        raise ValueError(f"{path}: RUC surface/pressure shapes differ")
    for name in required[1:] + optional:
        if name not in fields:
            continue
        target = (nzs,) + shape if name in ("TSLB", "SMOIS") else shape
        if fields[name].shape != target:
            raise ValueError(f"{path}: {name} expected shape {target}, got {fields[name].shape}")
    if np.any(fields.get("LAKEMASK", 0.) > 0.):
        raise ValueError(f"{path}: RUC lake-model points are not supported")
    if np.any(fields.get("SEAICE", fields.get("XICE", 0.)) > 0.):
        raise ValueError(f"{path}: RUC sea-ice points are not supported")
    # WRF's tables index these categories directly; never clip invalid input into
    # a different vegetation/soil class (the column API has separate safeguards).
    from gpuwrf.physics.ruclsm import load_ruc_tables, mminlu_to_ruc
    tab = load_ruc_tables(mminlu_to_ruc(attrs["MMINLU"]), table_dir)
    for name, count in (("IVGTYP", len(tab.z0tbl)), ("ISLTYP", len(tab.bb))):
        values = fields[name]
        if np.any(values != values.astype(np.int32)) or np.any((values < 1) | (values > count)):
            raise ValueError(f"{path}: invalid RUC {name} category (expected 1..{count})")
    static, land = build_ruc_bundles(
        fields, mminlu=attrs["MMINLU"], iswater=attrs["ISWATER"], isice=attrs["ISICE"],
        dt=float(dt), zs=zs, p1=p1, dtype="float32", run_dir=table_dir, tables=tab,
    )
    if np.any(ruc_unsupported_mask(land, static)):
        raise ValueError(f"{path}: unsupported RUC land columns")
    import jax.numpy as jnp
    # Preserve cold wrfinput values until the ktau=1 block runs on the first
    # actual physics input. In particular zero QVG is WRF's initialization flag.
    seeded = {"alb": albedo, "emiss": emiss, "qvg": fields.get("QVG", np.zeros(shape)),
              "qcg": fields.get("QCG", np.zeros(shape))}
    if "SOILT1" in fields:
        seeded["soilt1"] = fields["SOILT1"]
    if "QSFC" in fields:
        seeded["qsfc"] = fields["QSFC"]
    land = land._replace(**{name: jnp.asarray(value, jnp.float32) for name, value in seeded.items()})
    return static, land


def bind_ruc_land(namelist, run_dir, domain, *, num_soil_layers, state):
    """Shared daily/nested loader seam, called only when sf_surface_physics=3."""
    from gpuwrf.coupling.physics_couplers import _surface_column_view

    if int(namelist.mp_physics) != 8 or int(namelist.sf_sfclay_physics) != 5:
        raise ValueError("RUC CLI requires mp_physics=8 and sf_sfclay_physics=5")
    if int(namelist.sf_lake_physics) != 0:
        raise ValueError("RUC CLI does not support sf_lake_physics")
    view = _surface_column_view(state, namelist.grid)
    if view.psfc is None:
        raise ValueError("RUC CLI requires WRF hydrostatic surface pressure")
    # LSMRUC's argument is named p8w internally, but surface_driver passes
    # P_PHY (mass-level hydrostatic pressure), not its PSFC interface value.
    static, land = load_ruc_land(run_dir, domain, num_soil_layers=num_soil_layers,
                                 dt=namelist.dt_s, p1=view.p[..., 0])
    return replace(namelist, sf_surface_physics=3, use_noahmp=False,
                   ruc_static=static, ruc_land=land)
