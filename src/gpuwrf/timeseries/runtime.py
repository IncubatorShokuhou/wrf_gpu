"""Host setup/segment flush for the native single- and multi-domain pipeline."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .files import TimeSeriesWriter, read_tslist
from .locations import locate_stations


def require_supported_run(input_dir, *, restart=False, batch_size=1, replay=False):
    """Refuse unsupported modes before model/input initialization."""
    path = Path(input_dir) / "tslist"
    if not path.is_file():
        return
    if restart or batch_size != 1 or replay:
        raise ValueError("tslist currently requires a fresh native B=1 forecast; "
                         "checkpoint/resume, homogeneous batching and CPU-history replay are unsupported")


class TimeSeriesRuntime:
    def __init__(self, writers, max_root_steps):
        self.writers = writers
        self.max_root_steps = max_root_steps

    def flush(self, carries):
        """One bounded DtoH batch per domain at a host segment boundary."""
        import jax
        import jax.numpy as jnp
        result = dict(carries)
        for name, writer in self.writers.items():
            carry = carries[name]
            buf = carry.timeseries
            count, hours, surface, profiles = jax.device_get((buf.count, buf.hours, buf.surface, buf.profiles))
            count = int(count)
            if not 0 <= count <= len(hours):
                raise RuntimeError(f"{name}: time series buffer overflow ({count}/{len(hours)})")
            writer.write(hours[:count], surface[:count], profiles[:count])
            zero = jax.device_put(jnp.asarray(0, jnp.int32), device=buf.count.device)
            result[name] = carry.replace(timeseries=replace(buf, count=zero))
        return result

    def report(self):
        return {name: {"stations": len(w.locations), "rows_per_station": w.rows,
                       "levels": w.levels, "files": [str(w.path(loc, ending))
                       for loc in w.locations for ending in ("TS", "UU", "VV", "WW", "PH", "TH", "QV", "PR")]}
                for name, w in self.writers.items()}


def prepare_time_series(config, names, bundles, carries, dt_by_domain, run_start, wrf_namelist):
    """Return (runtime, seeded carries), or (None, original carries) without a tslist.

    At most one root step of child samples must fit. Tiny ts_buf_size values
    therefore use that minimum effective capacity on a child; all larger values
    are honored exactly. Segment boundaries may be more frequent than history
    writes, without changing the history alarms or numerical work.
    """
    path = Path(config.input_dir) / "tslist"
    if not path.is_file():
        return None, carries
    from gpuwrf.io.netcdf_lock import Dataset
    from .sampling import seed_buffer

    options = wrf_namelist.get("domains", {})
    solar = wrf_namelist.get("diags", {}).get("solar_diagnostics", 0)
    if isinstance(solar, (tuple, list)):
        solar = solar[0]
    if int(solar) != 0:
        raise ValueError("tslist solar_diagnostics requires WRF's extended solar format, which is unsupported")

    def scalar(key, default):
        value = options.get(key, default)
        if isinstance(value, (tuple, list)):
            value = value[0]
        return value

    limit = int(scalar("max_ts_locs", 5))
    capacity, levels = int(scalar("ts_buf_size", 200)), int(scalar("max_ts_level", 15))
    unstagger = bool(scalar("tslist_unstagger_winds", False))
    if capacity <= 0 or levels <= 0:
        raise ValueError("ts_buf_size and max_ts_level must be positive")
    stations = read_tslist(path, limit)
    if not stations:
        return None, carries
    # module_check_a_mundo.F caps this GLOBAL option against e_vert(1)-1,
    # even when a vertically refined child has more model levels.
    levels = min(levels, int(carries[names[0]].state.theta.shape[0]))
    sst_update = wrf_namelist.get("physics", {}).get("sst_update", 0)
    if isinstance(sst_update, (tuple, list)):
        sst_update = sst_update[0]
    root_dt = float(dt_by_domain[names[0]])
    plans = {}
    result = dict(carries)
    max_root_steps = None
    for name in names:
        carry = carries[name]
        if (carry.noahmp_land is None or carry.history_diagnostics is None
                or carry.radiation_diagnostics is None):
            raise ValueError(f"{name}: tslist requires Noah-MP and held in-step surface/radiation diagnostics")
        if int(bundles[name].namelist.sf_surface_physics) != 4:
            raise ValueError(f"{name}: tslist currently supports sf_surface_physics=4")
        nz, ny, nx = carry.state.theta.shape
        if nz < 2:
            raise ValueError(f"{name}: WRF calc_p8w requires at least two mass levels")
        profile_levels = min(levels, nz)  # WRF check_a_mundo clamps max_ts_level.
        with Dataset(Path(config.input_dir) / f"wrfinput_{name}") as ds:
            attrs = {key: ds.getncattr(key) for key in ds.ncattrs()}
            locations = locate_stations(stations, int(name[1:]), ds['XLAT'][0], ds['XLONG'][0],
                                        ds['HGT'][0], attrs)
            if not locations:
                continue
            if unstagger and any(loc.i == nx or loc.j == ny for loc in locations):
                raise ValueError(f"{name}: unstaggered tslist at the high edge requires unavailable rotation halos")
            land = [float(ds['LANDMASK'][0, loc.j-1, loc.i-1]) > .5 for loc in locations]
        min_capacity = int(round(root_dt / float(dt_by_domain[name])))
        effective_capacity = max(capacity, min_capacity)
        steps = max(1, int(np.floor(effective_capacity * float(dt_by_domain[name])/root_dt + 1.e-9)))
        max_root_steps = steps if max_root_steps is None else min(steps, max_root_steps)
        import jax
        buffer = seed_buffer(
            locations, profile_levels, effective_capacity, land=land,
            unstagger=unstagger, sst_update=bool(sst_update))
        # Match the seeded operational carry's committed placement from call 1.
        # An uncommitted observer seed otherwise causes a second whole-model JIT.
        result[name] = carry.replace(timeseries=jax.device_put(buffer, device=carry.state.theta.device))
        plans[name] = locations, profile_levels
    if not plans:
        return None, carries
    # Validate all domains before opening any output files.
    writers = {name: TimeSeriesWriter(config.output_dir, locs, run_start.strftime('%Y-%m-%d_%H:%M:%S'), lev)
               for name, (locs, lev) in plans.items()}
    return TimeSeriesRuntime(writers, max_root_steps or 1), result
