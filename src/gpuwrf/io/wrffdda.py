"""Read ``wrffdda_d0N`` (WRF auxinput10, real.exe ``grid_fdda >= 1``) for grid nudging.

real.exe writes one record per analysis time t_n (main/real_em.F:904-1110,1246):
``*_NDG_OLD`` = analysis(t_n), ``*_NDG_NEW`` = analysis(t_n+1), ``T_NDG`` = dry
``th_phy - t0``.  wrf.exe reads the next record at every auxinput10 alarm,
``RingTime = start``, ``RingInterval = gfdda_interval_m``, ``StopTime = start +
gfdda_end_h`` (frame/module_domain.F:2318).  The port keeps the records the run
reads on device and activates record r at the first step whose entry time is at
or after its alarm time (the same rule as the auxinput4 reader).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwrf.io.fdda_namelist import grid_fdda_options, grid_fdda_code, wrffdda_filename

_FIELDS = (("u", "U_NDG"), ("v", "V_NDG"), ("t", "T_NDG"), ("q", "Q_NDG"))


def read_times(path: Path) -> list[datetime]:
    from netCDF4 import chartostring
    from gpuwrf.io.netcdf_lock import Dataset

    with Dataset(path, "r") as ds:
        return [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S") for t in chartostring(ds["Times"][:])]


def load_grid_fdda(run_dir, wrf_namelist: Mapping[str, Any], domain: str, *, run_start: datetime,
                   dt_s: float, nz: int, ny: int, nx: int, run_seconds: float | None = None,
                   dx_m: float | None = None):
    """Return ``GridFddaInputs`` for ``domain`` or None when grid_fdda=0.

    Raises ``ValueError`` for unsupported selections, a missing/short file, record times
    off the analysis interval, shape mismatches or non-finite analyses.
    """

    options = grid_fdda_options(wrf_namelist, domain, dt_s)
    if options is None:
        return None
    import jax.numpy as jnp
    from gpuwrf.io.netcdf_lock import Dataset

    from gpuwrf.assimilation.fdda_grid import GridFddaInputs, GridFddaOptions
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs, SpectralFddaOptions

    spectral = grid_fdda_code(wrf_namelist, domain) == 2

    path = Path(run_dir) / wrffdda_filename(wrf_namelist, domain)
    if not path.is_file():
        raise ValueError(f"{domain}: grid_fdda={2 if spectral else 1} needs the real.exe analysis file {path} (auxinput10)")
    interval_s = int(options["interval_m"]) * 60
    end_s = int(options["end_h"]) * 3600 if int(options["end_h"]) > 0 else None  # 0 -> stream never stops

    def read_by_run(t: float) -> bool:
        """The alarm at t rings (t <= StopTime) and a step entry at/after t exists in this run."""

        if end_s is not None and t > end_s:
            return False
        return run_seconds is None or np.ceil(t / float(dt_s)) * float(dt_s) < float(run_seconds)

    start = run_start.replace(tzinfo=None)
    with Dataset(path, "r") as ds:
        from netCDF4 import chartostring

        times = [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S") for t in chartostring(ds["Times"][:])]
        offsets = np.array([(t - start).total_seconds() for t in times])
        if offsets.size == 0 or offsets[0] != 0:
            raise ValueError(f"{path}: the first wrffdda record must be valid at the run start {start}")
        expected = np.arange(offsets.size) * interval_s
        if not np.array_equal(offsets, expected):
            raise ValueError(f"{path}: record times {offsets.tolist()} s are not the gfdda_interval_m="
                             f"{options['interval_m']} sequence (WRF reads auxinput10 sequentially)")
        wanted = np.array([t for t in expected if read_by_run(t)])
        if read_by_run(float(offsets[-1] + interval_s)) and (end_s is not None or run_seconds is not None):
            raise ValueError(f"{path}: has records up to {offsets[-1] / 3600:g} h but the run reads the "
                             f"auxinput10 record at {(offsets[-1] + interval_s) / 3600:g} h")
        shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "t": (nz, ny, nx), "q": (nz, ny, nx)}
        fields = _FIELDS
        if spectral:
            shapes["ph"] = (nz + 1, ny, nx)
            fields += (("ph", "PH_NDG"),)
        arrays: dict[str, Any] = {}
        for name, var in fields:
            for suffix in ("OLD", "NEW"):
                if f"{var}_{suffix}" not in ds.variables:
                    raise ValueError(f"{path}: missing {var}_{suffix} required by grid_fdda={2 if spectral else 1}")
                raw = np.ma.asarray(ds[f"{var}_{suffix}"][: wanted.size])
                if np.ma.getmaskarray(raw).any():
                    raise ValueError(f"{path}: {var}_{suffix} contains missing data")
                a = np.asarray(raw, dtype=np.float32)
                if spectral and name in ("u", "v", "ph") and a.shape == (wanted.size, nz, ny, nx):
                    # WRF's packed fdda3d I/O can serialize every species on
                    # mass dimensions despite the U/V/PH member stagger tags.
                    # The spectral caller only reads this mass footprint; pad
                    # the unused east/north/top slot for the public State layout.
                    padding = [(0, 0)] * 4
                    padding[{"u": 3, "v": 2, "ph": 1}[name]] = (0, 1)
                    a = np.pad(a, padding)
                if a.shape != (wanted.size, *shapes[name]) or not np.isfinite(a).all():
                    raise ValueError(f"{path}: {var}_{suffix} must be finite {(wanted.size, *shapes[name])}, "
                                     f"got {a.shape}")
                arrays[f"{name}_{suffix.lower()}"] = jnp.asarray(a)
    steps = tuple(int(s) for s in np.ceil(wanted / float(dt_s)).astype(np.int64))
    if spectral:
        return SpectralFddaInputs(**arrays, options=SpectralFddaOptions(**options), record_steps=steps)
    base = GridFddaInputs(**arrays, options=GridFddaOptions(**options), record_steps=steps)
    from gpuwrf.io.wrfsfdda import surface_code, attach_surface_fdda
    if surface_code(wrf_namelist,domain):
        if dx_m is None:raise ValueError(f'{domain}: surface FDDA requires the domain dx_m')
        return attach_surface_fdda(base,run_dir,wrf_namelist,domain,run_start=run_start,dt_s=dt_s,
            ny=ny,nx=nx,dx_m=float(dx_m),run_seconds=run_seconds)
    return base
