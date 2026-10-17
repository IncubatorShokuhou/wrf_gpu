"""Read-only native WRF case preflight. See docs/doctor.md for definitions."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from .land import category_issues
from .common import char_times, finite_number, parse_namelist, run_window, support_report, value

NEXT_TIME = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
SOURCE_KEYS = ("SOURCE", "IC_SOURCE", "BC_SOURCE", "METGRID_SOURCE", "SOURCE_MODEL", "MMINLU")


def field(ds, name, record=0):
    var = ds.variables[name]
    raw = var[record] if var.dimensions and var.dimensions[0] == "Time" else var[:]
    data = np.asarray(np.ma.filled(np.ma.asarray(raw).astype(np.float64), np.nan))
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains missing/NaN/Inf values")
    return data


def scan_numeric(ds):
    """Record-wise scan: reject nonfinite input even in unused fields."""
    for name, var in ds.variables.items():
        if np.dtype(var.dtype).kind not in "fiu":
            continue
        records = range(var.shape[0]) if var.dimensions and var.dimensions[0] == "Time" else (0,)
        for record in records:
            field(ds, name, record)


def land_table(path, section):
    """Read actual section header and labelled water categories in LANDUSE.TBL."""
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == section:
            count = int(lines[index + 1].split(",")[0])
            water = set()
            for row in lines[index + 2:]:
                match = re.match(r"\s*(\d+)\s*,.*['\"]([^'\"]+)['\"]", row)
                if match:
                    if match.group(2).strip().lower() in {"water", "water bodies"}:
                        water.add(int(match.group(1)))
                elif row.strip() not in {"SUMMER", "WINTER", ""}:
                    break
            return count, water
    raise ValueError(f"LANDUSE.TBL has no {section!r} section")


def vram_estimate(cells):
    """Advisory interpolation of PROD A58; allocator reserves are separate."""
    prod_cells = (120 * 70 + 267 * 117) * 44
    # A26/A38/A58: empirical configuration-specific overhead is substantial.
    return {"label": "INFERRED, not an admission limit", "grid_cells": cells,
            "live_gib": 4.00e9 / 2**30 * cells / prod_cells,
            "planning_gib": (963 * 2**20 + 5.61e9 * cells / prod_cells) / 2**30,
            "calibration": "FINDINGS A26/A38/A58 (LW6/LW9 PROD/WN3); PROD A58 live=4.00 GB, C-auto=5.61 GB; WN3 external=963 MiB",
            "limitations": "scheme, nests, output, compiler, allocator and GPU dependent; no capacity guarantee"}


def validate_boundary_axes(ds, ic_dims, nl):
    """WRF base/tendency strips retain each variable's C-grid staggering."""
    width = finite_number(value(nl, "spec_bdy_width", default=5), "spec_bdy_width", positive=True)
    if width != int(width) or "bdy_width" not in ds.dimensions or len(ds.dimensions["bdy_width"]) != width:
        raise ValueError("wrfbdy bdy_width must match namelist spec_bdy_width (WRF default 5)")
    if width > min(ic_dims.get("west_east", 0), ic_dims.get("south_north", 0)):
        raise ValueError("boundary width exceeds the IC horizontal grid")
    for key in ("U", "V", "T", "PH", "MU", "QVAPOR", "W"):
        # W is optional in the native boundary loader; validate it when supplied.
        if key == "W" and not any(name.startswith("W_B") for name in ds.variables):
            continue
        for suffix in ("BXS", "BXE", "BYS", "BYE", "BTXS", "BTXE", "BTYS", "BTYE"):
            name = f"{key}_{suffix}"
            if name not in ds.variables:
                raise ValueError(f"missing boundary field {name}")
            along = ("south_north_stag" if key == "V" else "south_north") if suffix.endswith(("XS", "XE")) else ("west_east_stag" if key == "U" else "west_east")
            vertical = () if key == "MU" else ("bottom_top_stag" if key in {"PH", "W"} else "bottom_top",)
            expected = ("Time", "bdy_width", *vertical, along)
            if ds.variables[name].dimensions != expected:
                raise ValueError(f"boundary {name} dimensions must be {expected}")
            for axis in (*vertical, along):
                if axis not in ic_dims or len(ds.dimensions[axis]) != ic_dims[axis]:
                    raise ValueError(f"boundary {name} {axis} length differs from wrfinput")


def inspect_case(case_dir, namelist=None):
    case = Path(case_dir)
    namelist = Path(namelist) if namelist else case / "namelist.input"
    report = {"schema": "GpuwrfDoctor", "schema_version": 1, "case_dir": str(case.absolute()),
              "issues": [], "domains": [], "checks": "native real.exe inputs; read only"}

    def issue(code, message, severity="ERROR", file=None):
        report["issues"].append(dict(code=code, severity=severity, message=str(message), file=file))

    try:
        nl = parse_namelist(namelist)
        start, end = run_window(nl)
        ndom = int(value(nl, "max_dom", default=1))
        if not 1 <= ndom <= 99:
            raise ValueError("max_dom must be 1..99")
        interval = finite_number(value(nl, "interval_seconds", default=21600), "interval_seconds", positive=True)
        dt = finite_number(value(nl, "time_step"), "time_step", positive=True)
        denominator = finite_number(value(nl, "time_step_fract_den", default=1), "time_step_fract_den", positive=True)
        dt += finite_number(value(nl, "time_step_fract_num", default=0), "time_step_fract_num") / denominator
        if dt <= 0:
            raise ValueError("effective time step must be positive")
        support = support_report(nl)
        report.update(run_start=start.isoformat(), run_end=end.isoformat(), support=support)
        for error in support["errors"]:
            issue("NAMELIST_SUPPORT", error)
        for warning in support["warnings"]:
            issue("NAMELIST_SUPPORT", warning, "WARNING")
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        issue("NAMELIST", exc)
        report["status"] = "FAIL"
        return report

    cells = 0
    metadata = {}
    dimensions = {}
    for domain in range(ndom):
        name = f"d{domain+1:02d}"
        path = case / f"wrfinput_{name}"
        try:
            with Dataset(path) as ds:
                scan_numeric(ds)
                times = char_times(ds)
                if times[0] != start or len(times) != 1:
                    issue("IC_TIME", f"expected one IC record at {start}; found {times}", file=path.name)
                dims = {key: len(dim) for key, dim in ds.dimensions.items()}
                dimensions[name] = dims
                for dim, key in (("west_east", "e_we"), ("south_north", "e_sn"), ("bottom_top", "e_vert")):
                    if dim not in dims or dims[dim] < 1:
                        raise ValueError(f"missing/empty dimension {dim}")
                    requested = value(nl, key, domain)
                    if requested is not None and dims[dim] != int(requested) - 1:
                        issue("GRID_DIMENSION", f"{dim}={dims[dim]} vs namelist {key}={requested} (mass dimension must be {key}-1)", file=path.name)
                    if dim + "_stag" in dims and dims[dim + "_stag"] != dims[dim] + 1:
                        issue("STAGGER_DIMENSION", f"{dim}_stag must equal {dim}+1", file=path.name)
                cells += dims["west_east"] * dims["south_north"] * dims["bottom_top"]
                metadata[name] = {key: str(ds.getncattr(key)).strip() for key in SOURCE_KEYS if key in ds.ncattrs()}
                required = ("LANDMASK", "XLAND", "LU_INDEX", "ISLTYP", "IVGTYP", "HGT", "PH", "PHB", "W",
                            "U", "V", "T", "P", "PB", "MU", "MUB", "QVAPOR")
                missing = [key for key in required if key not in ds.variables]
                if missing:
                    raise ValueError(f"missing required fields: {', '.join(missing)}")
                xy = ("south_north", "west_east")
                xyz = ("bottom_top", *xy)
                shapes = {key: xyz for key in ("T", "P", "PB", "QVAPOR")}
                shapes.update({key: xy for key in ("MU", "MUB", "HGT", "LANDMASK", "XLAND", "LU_INDEX", "ISLTYP", "IVGTYP")})
                shapes.update(U=("bottom_top", "south_north", "west_east_stag"),
                              V=("bottom_top", "south_north_stag", "west_east"))
                shapes.update({key: ("bottom_top_stag", *xy) for key in ("PH", "PHB", "W")})
                for key, spatial in shapes.items():
                    if ds.variables[key].dimensions != ("Time", *spatial):
                        issue("FIELD_DIMENSION", f"{key} must have dimensions {('Time', *spatial)}", file=path.name)
                lm, xl, lu, soil, veg = [field(ds, k) for k in ("LANDMASK", "XLAND", "LU_INDEX", "ISLTYP", "IVGTYP")]
                if not np.isin(lm, (0, 1)).all() or not np.array_equal(xl, 2 - lm):
                    issue("LANDMASK_XLAND", "LANDMASK must be 1 land/0 water and XLAND must equal 2-LANDMASK", file=path.name)
                ncat = float(getattr(ds, "NUM_LAND_CAT", value(nl, "num_land_cat", default=24)))
                declared = value(nl, "num_land_cat")
                if declared is not None and int(declared) != ncat:
                    issue("LAND_CATEGORIES", f"NUM_LAND_CAT={ncat} vs num_land_cat={declared}", file=path.name)
                nsoil = int(getattr(ds, "NUM_SOIL_CAT", value(nl, "num_soil_cat", default=16)))
                for code, message, severity in category_issues(ds, ncat, value(nl, "sf_urban_physics", domain, 0), bool(value(nl, "use_wudapt_lcz", default=False))):
                    issue(code, message, severity, file=path.name)
                for key, data, count in (("ISLTYP", soil, nsoil),):
                    if ((data < 1) | (data > count) | (data != np.floor(data))).any():
                        issue("CATEGORY_RANGE", f"{key} must be integer in 1..{count}", file=path.name)
                if not np.array_equal(lu, veg):
                    issue("LU_IVGTYP", "LU_INDEX differs from IVGTYP", "WARNING", path.name)
                table = case / "LANDUSE.TBL"
                water = {int(ds.ISWATER)} if "ISWATER" in ds.ncattrs() else set()
                if table.is_file():
                    count, table_water = land_table(table, str(getattr(ds, "MMINLU", "")).strip())
                    if ncat > count:
                        issue("LU_TABLE", f"NUM_LAND_CAT={ncat} exceeds table section count {count}", file=path.name)
                    if water and table_water and not water <= table_water:
                        issue("LU_TABLE", f"ISWATER={water} differs from table water categories {table_water}", file=path.name)
                    water |= table_water
                else:
                    issue("LU_TABLE_UNKNOWN", "no case LANDUSE.TBL; category bounds checked, table compatibility unknown", "WARNING", path.name)
                # Lakes may be a separate category, identified explicitly by WRF.
                if "ISLAKE" in ds.ncattrs() and int(ds.ISLAKE) > 0:
                    water.add(int(ds.ISLAKE))
                if water and not np.array_equal(np.isin(lu, sorted(water)), lm == 0):
                    issue("LU_LANDMASK", "land-use water/lake categories disagree with LANDMASK", file=path.name)
                expected_soil = int(value(nl, "num_soil_layers", default=4))
                if int(value(nl, "sf_surface_physics", domain, 4)) != 0:
                    if dims.get("soil_layers_stag") != expected_soil:
                        issue("SOIL_LEVELS", f"soil_layers_stag={dims.get('soil_layers_stag')} vs num_soil_layers={expected_soil}", file=path.name)
                    for key in ("TSLB", "SMOIS", "SH2O"):
                        if key not in ds.variables or "soil_layers_stag" not in ds.variables[key].dimensions:
                            issue("SOIL_FIELDS", f"{key} missing or lacks soil_layers_stag", file=path.name)
                for key, low, high, units in (("HGT", -500, 10000, {"m"}), ("TSLB", 150, 400, {"K"}),
                                               ("SST", 200, 350, {"K"}), ("SMOIS", 0, 1, {"m3 m-3", "m^3/m^3", "m3/m3"}),
                                               ("QVAPOR", 0, .2, {"kg kg-1", "kg/kg"})):
                    if key not in ds.variables:
                        continue
                    data = field(ds, key)
                    active = data[lm == 0] if key == "SST" else data.ravel()
                    if active.size and ((active < low) | (active > high)).any():
                        issue("FIELD_RANGE", f"{key} outside broad sanity range {low}..{high}", file=path.name)
                    actual_units = str(getattr(ds.variables[key], "units", "")).strip()
                    if actual_units not in units:
                        issue("FIELD_UNITS", f"{key} units {actual_units!r}; expected one of {sorted(units)}", "WARNING", path.name)
                dx = finite_number(getattr(ds, "DX", value(nl, "dx", domain)), "DX", positive=True)
                dy = finite_number(getattr(ds, "DY", value(nl, "dy", domain, dx)), "DY", positive=True)
                domain_dt = dt
                parent = domain
                seen = set()
                while parent > 0:
                    if parent in seen:
                        raise ValueError("cyclic parent_id")
                    seen.add(parent)
                    domain_dt /= finite_number(value(nl, "parent_time_step_ratio", parent, 1), "parent_time_step_ratio", positive=True)
                    parent_id = int(value(nl, "parent_id", parent, 1))
                    if not 1 <= parent_id <= parent:
                        raise ValueError("child parent_id must identify an earlier domain")
                    parent = parent_id - 1
                hgt = field(ds, "HGT")
                slope = max([0.] + [float(np.max(np.abs(np.diff(hgt, axis=axis)))) / spacing
                                    for axis, spacing in ((0, dy), (1, dx)) if hgt.shape[axis] > 1])
                dz = np.diff((field(ds, "PH") + field(ds, "PHB")) / 9.80665, axis=0)
                if (dz <= 0).any():
                    raise ValueError("PH+PHB must increase vertically (positive layer thickness)")
                w = field(ds, "W")
                vertical_cfl = float(np.max(np.maximum(abs(w[:-1]), abs(w[1:])) * domain_dt / dz))
                advice = {"dt_s": domain_dt, "typical_dt_s": 6 * min(dx, dy) / 1000,
                          "max_adjacent_terrain_slope": slope, "initial_vertical_cfl": vertical_cfl,
                          "scope": "initial physical W only; excludes contravariant terrain velocity and acoustic CFL; advisory"}
                from gpuwrf.io.windfarm import windfarm_option
                if windfarm_option(nl,name) == 1:
                    from gpuwrf.io.windfarm import read_windfarm
                    from gpuwrf.io.obs_projection import ObservationProjection
                    from .windfarm import density_advice
                    projection = None
                    if value(nl,'windfarm_ij',default=0) == 0:
                        projection = ObservationProjection(int(ds.MAP_PROJ),float(ds['XLAT'][0,0,0]),float(ds['XLONG'][0,0,0]),
                                  float(ds.TRUELAT1),float(ds.TRUELAT2),float(ds.STAND_LON),dx,dy)
                    farm = read_windfarm(case,ij=value(nl,'windfarm_ij',default=0),nx=dims['west_east'],ny=dims['south_north'],
                                         projection=projection,tke_factor=value(nl,'windfarm_tke_factor',default=.25))
                    u,v = field(ds,'U'),field(ds,'V')
                    speed = np.sqrt((.5*(u[:,:,:-1]+u[:,:,1:]))**2+(.5*(v[:,:-1,:]+v[:,1:,:]))**2)
                    wind_advice = density_advice(farm,dx=dx,dt=domain_dt,min_rotor_dz=float(dz.min()),max_speed=float(speed.max()))
                    advice['windfarm'] = wind_advice
                    if wind_advice['warning']:
                        issue('WINDFARM_DRAG_ADVICE',f'{wind_advice}; consider reducing dt or turbine density','WARNING',path.name)
                if domain_dt > advice["typical_dt_s"] or slope > .5 or vertical_cfl > 1:
                    issue("CFL_ADVICE", f"{advice}; consider reducing dt in steep terrain; no stability guarantee", "WARNING", path.name)
                report["domains"].append(dict(domain=name, dimensions=dims, advice=advice))
        except (OSError, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
            issue("INPUT_INVALID", exc, file=path.name)

    for domain in range(ndom):
        name = f"d{domain+1:02d}"
        kinds = (["wrfbdy"] if domain == 0 else []) + (["wrflowinp"] if value(nl, "sst_update", default=0) else [])
        for kind in kinds:
            path = case / f"{kind}_{name}"
            try:
                with Dataset(path) as ds:
                    scan_numeric(ds)
                    times = char_times(ds)
                    if (times[0] != start if kind == "wrfbdy" else times[0] > start):
                        raise ValueError(f"{kind} first record does not cover IC start {start}")
                    if kind == "wrfbdy":
                        validate_boundary_axes(ds, dimensions.get(name, {}), nl)
                        if any((b-a).total_seconds() != interval for a,b in zip(times,times[1:])):
                            raise ValueError("boundary Times spacing differs from interval_seconds")
                        coverage = times[-1] + timedelta(seconds=interval)
                        if NEXT_TIME in ds.variables:
                            next_times = char_times(ds, NEXT_TIME)
                            if len(next_times) != len(times) or any(b != a + timedelta(seconds=interval) for a,b in zip(times,next_times)):
                                raise ValueError("next-boundary times disagree with boundary records/interval")
                            coverage = min(coverage, next_times[-1])
                    else:
                        if "SST" not in ds.variables:
                            raise ValueError("sst_update=1 requires SST in wrflowinp")
                        if ds.variables["SST"].dimensions != ("Time", "south_north", "west_east"):
                            raise ValueError("wrflowinp SST must have Time,south_north,west_east dimensions")
                        coverage = times[-1]  # no extrapolation beyond last SST record
                    if coverage < end:
                        issue("FORCING_COVERAGE", f"{kind} ends {coverage}, requested end {end}", file=path.name)
                    for key, dim in ds.dimensions.items():
                        if key not in {"Time", "DateStrLen"} and key in dimensions.get(name, {}) and len(dim) != dimensions[name][key]:
                            issue("FORCING_DIMENSION", f"{key} differs from wrfinput", file=path.name)
                    for key in SOURCE_KEYS:
                        if key in ds.ncattrs() and key in metadata.get(name, {}) and str(ds.getncattr(key)).strip() != metadata[name][key]:
                            issue("SOURCE_MISMATCH", f"IC/forcing {key} differ: {metadata[name][key]!r} vs {str(ds.getncattr(key))!r}; heuristic only", "WARNING", path.name)
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
                issue("FORCING_INVALID", exc, file=path.name)
    report["vram"] = (vram_estimate(cells) if len(report["domains"]) == ndom else
                      {"label": "UNKNOWN", "reason": "complete valid grid inventory unavailable"})
    report["status"] = "FAIL" if any(i["severity"] == "ERROR" for i in report["issues"]) else "PASS"
    return report
