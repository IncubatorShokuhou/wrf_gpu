"""Single-root WPS Mercator domains and explicitly inferred memory budgets.

The sphere (6370000 m) and Mercator formulas match WPS module_map_utils.F.
The bbox is an area of interest, surrounded by 12 grid cells on each side.
Memory is a conservative planning estimate, never an allocator admission proof.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

RADIUS_M = 6370000.0
GIB = 1024**3
# FINDINGS A58: PROD two-domain live peak 4.00 GB, 44 mass levels.
BYTES_PER_CELL_LEVEL = 4_000_000_000 / ((120 * 70 + 267 * 117) * 44)


def memory_gib(nx: int, ny: int, nz: int = 44) -> float:
    """[I] 25% margin over scaled A58 live memory + 1 GiB context/modules."""
    return 1.25 * nx * ny * nz * BYTES_PER_CELL_LEVEL / GIB + 1.0


def profiles() -> list[dict]:
    """[I] 44-level square mass grids, keeping 2 GiB outside our estimate."""
    rows = []
    for capacity in (12, 16, 24, 32):
        side = math.floor(math.sqrt((capacity - 3) * GIB / (1.25 * 44 * BYTES_PER_CELL_LEVEL)))
        rows.append(dict(vram_gib=capacity, square_mass_grid=side,
                         estimated_gib=memory_gib(side, side), label="I",
                         measured_limit=False))
    return rows


def _merc_y(lat: float, scale: float) -> float:
    return scale * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _merc_lat(y: float, scale: float) -> float:
    return math.degrees(2 * math.atan(math.exp(y / scale)) - math.pi / 2)


def plan_region(bbox: list[float], resolution_km: float, date: str,
                hours: int, vram_gib: int = 12) -> dict:
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("bbox must contain finite WEST SOUTH EAST NORTH degrees")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -70 <= south < north <= 70):
        raise ValueError("bbox requires WEST < EAST, SOUTH < NORTH, latitudes within ±70; no dateline crossing")
    if east - west > 40 or north - south > 30:
        raise ValueError("quickstart supports regional extents up to 40° longitude / 30° latitude")
    if not math.isfinite(resolution_km) or not 1 <= resolution_km <= 30:
        raise ValueError("resolution must be finite, from 1 to 30 km")
    if hours < 3 or hours > 120 or hours % 3:
        raise ValueError("hours must be a multiple of 3, from 3 to 120 (3-hour forcing)")
    if vram_gib not in (12, 16, 24, 32):
        raise ValueError("vram-gib must be 12, 16, 24 or 32")
    try:
        start = datetime.strptime(date, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DDTHH in UTC") from exc
    if start.hour not in (0, 6, 12, 18):
        raise ValueError("GFS cycles start at 00, 06, 12 or 18 UTC")
    dx = resolution_km * 1000
    truelat = (south + north) / 2
    scale = RADIUS_M * math.cos(math.radians(truelat))
    y0, y1 = _merc_y(south, scale), _merc_y(north, scale)
    ref_y = (y0 + y1) / 2
    nx = max(40, math.ceil(scale * math.radians(east - west) / dx) + 24)
    ny = max(40, math.ceil((y1 - y0) / dx) + 24)
    estimate = memory_gib(nx, ny)
    if estimate > vram_gib - 2:
        raise ValueError(f"[I] grid {nx}×{ny}×44 estimates {estimate:.2f} GiB plus 2 GiB reserve; "
                         f"exceeds {vram_gib} GiB profile. Increase spacing or reduce bbox.")
    # Align every hour and radiation alarm; never discover nonintegral steps after init.
    dt = max(d for d in range(1, int(6 * resolution_km) + 1) if 3600 % d == 0 and 600 % d == 0)
    lon = (west + east) / 2
    bounds = [lon - math.degrees(nx * dx / (2 * scale)),
              _merc_lat(ref_y - ny * dx / 2, scale),
              lon + math.degrees(nx * dx / (2 * scale)),
              _merc_lat(ref_y + ny * dx / 2, scale)]
    if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -75 or bounds[3] > 75:
        raise ValueError("padded domain crosses the supported longitude/latitude boundary")
    return dict(schema="GpuwrfRegionPlan", schema_version=1, bbox=bbox,
                domain_bbox=bounds, resolution_km=resolution_km, nx=nx, ny=ny,
                e_we=nx + 1, e_sn=ny + 1, e_vert=45, dx_m=dx, time_step=dt,
                map_proj="mercator", truelat1=truelat,
                ref_lat=_merc_lat(ref_y, scale), ref_lon=lon,
                start=start.strftime("%Y-%m-%d_%H:%M:%S"),
                end=(start + timedelta(hours=hours)).strftime("%Y-%m-%d_%H:%M:%S"),
                hours=hours, boundary_hours=list(range(0, hours + 1, 3)),
                vram_gib=vram_gib, memory_estimate_gib=estimate,
                memory_label="I", gpu_qualified=False,
                cu_physics=1 if resolution_km >= 9 else 0)


def namelists(plan: dict, geog: Path, num_metgrid_levels: int = 34) -> tuple[str, str]:
    """Return WPS / real namelists; metgrid's actual levels replace the placeholder."""
    if "'" in str(geog) or "\n" in str(geog):
        raise ValueError("geog path cannot contain quotes or newlines")
    wps = f"""&share
 wrf_core='ARW', max_dom=1,
 start_date='{plan['start']}', end_date='{plan['end']}',
 interval_seconds=10800, io_form_geogrid=2,
/
&geogrid
 parent_id=1, parent_grid_ratio=1, i_parent_start=1, j_parent_start=1,
 e_we={plan['e_we']}, e_sn={plan['e_sn']},
 geog_data_res='lowres', dx={plan['dx_m']}, dy={plan['dx_m']},
 map_proj='mercator', ref_lat={plan['ref_lat']:.10f}, ref_lon={plan['ref_lon']:.10f},
 truelat1={plan['truelat1']:.10f}, stand_lon={plan['ref_lon']:.10f},
 geog_data_path='{geog}',
/
&ungrib
 out_format='WPS', prefix='FILE',
/
&metgrid
 fg_name='FILE', io_form_metgrid=2,
/
"""
    start = datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S")
    end = datetime.strptime(plan["end"], "%Y-%m-%d_%H:%M:%S")
    time_fields = "\n".join(f" {prefix}_{key}={getattr(value, key)}," for prefix, value in
                            (("start", start), ("end", end)) for key in
                            ("year", "month", "day", "hour", "minute", "second"))
    wrf = f"""&time_control
 run_days=0, run_hours={plan['hours']},
{time_fields}
 interval_seconds=10800, input_from_file=.true.,
 history_interval=60, frames_per_outfile=1, restart=.false.,
 io_form_history=2, io_form_restart=2, io_form_input=2, io_form_boundary=2,
/
&domains
 time_step={plan['time_step']}, max_dom=1,
 e_we={plan['e_we']}, e_sn={plan['e_sn']}, e_vert=45,
 dx={plan['dx_m']}, dy={plan['dx_m']}, p_top_requested=5000,
 num_metgrid_levels={num_metgrid_levels}, num_metgrid_soil_levels=4,
 grid_id=1, parent_id=1, i_parent_start=1, j_parent_start=1,
 parent_grid_ratio=1, parent_time_step_ratio=1, feedback=0,
/
&physics
 mp_physics=8, ra_lw_physics=4, ra_sw_physics=4, radt=10,
 sf_sfclay_physics=5, sf_surface_physics=4, bl_pbl_physics=5, bldt=0,
 cu_physics={plan['cu_physics']}, cudt=0,
 isfflx=1, ifsnow=1, icloud=1, surface_input_source=1,
 num_soil_layers=4, num_land_cat=21, sf_urban_physics=0,
/
&dynamics
 w_damping=1, diff_opt=1, km_opt=4, diff_6th_opt=2, diff_6th_factor=0.12,
 base_temp=290., damp_opt=3, zdamp=5000., dampcoef=0.2,
 non_hydrostatic=.true., moist_adv_opt=1, scalar_adv_opt=1,
 epssm=0.5, gwd_opt=0,
/
&bdy_control
 spec_bdy_width=5, spec_zone=1, relax_zone=4, specified=.true., nested=.false.,
/
&namelist_quilt
 nio_tasks_per_group=0, nio_groups=1,
/
"""
    # The public gpuwrf namelist reader expects one assignment per line.
    wrf = re.sub(r", (?=[a-z][a-z_0-9]*\s*=)", ",\n ", wrf)
    return wps, wrf
