"""Bounded asynchronous publication of compact host-only NetCDF frames."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import os
from pathlib import Path
import tempfile

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from .config import FIELDS


def write_frames(directory, spec, points, lat, lon, ids, dt, run_start, steps, values, journal=None):
    """Publish a complete immutable chunk; foreign paths are never overwritten."""
    if (not np.isfinite(values).all() or len(steps) == 0 or
            np.any(np.diff(steps) <= 0) or np.any(steps % spec.steps(dt))):
        raise ValueError("regional stream contains NaN/Inf or invalid frame clocks")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"wrfout_stream_{spec.name}_{spec.domain}_{int(steps[0]):010d}_{int(steps[-1]):010d}.nc"
    if os.path.lexists(target):
        raise FileExistsError(f"regional stream target exists: {target}")
    fd, name = tempfile.mkstemp(prefix=".regional-", suffix=".nc", dir=directory)
    os.close(fd)
    temporary = Path(name)
    try:
        with Dataset(temporary, "w", format="NETCDF4") as ds:
            ds.createDimension("Time", len(steps))
            ds.createDimension("point", len(ids))
            ds.createDimension("bounds", 2)
            ds.createDimension("DateStrLen", 19)
            ds.Conventions = "CF-1.8"
            ds.gpuwrf_stream_schema = 1
            ds.stream_name = spec.name
            ds.domain = spec.domain
            ds.configuration_sha256 = spec.signature
            ds.sampling = "post-solve own-step samples in (window_start, window_end]; no spatial averaging"
            time = ds.createVariable("time", "f8", ("Time",))
            time.units = "seconds since " + run_start.strftime("%Y-%m-%d %H:%M:%S")
            time.calendar = "standard"
            time.bounds = "time_bounds"
            time[:] = steps.astype(np.float64)*dt
            ds.createVariable("time_bounds", "f8", ("Time", "bounds"))[:] = np.column_stack((steps-spec.steps(dt), steps))*dt
            stamps = [(run_start+timedelta(seconds=int(step)*dt)).strftime("%Y-%m-%d_%H:%M:%S") for step in steps]
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t) for t in stamps], "S1")
            ds.createVariable("point_id", str, ("point",))[:] = np.asarray(ids, object)
            y, x = points
            for field, data, units in (("XLAT", lat, "degrees_north"), ("XLONG", lon, "degrees_east"),
                                       ("i", x+1, "1"), ("j", y+1, "1")):
                variable = ds.createVariable(field, "i4" if field in {"i", "j"} else "f4", ("point",))
                variable.units = units
                variable[:] = data
            for k, (field, statistic) in enumerate(spec.channels):
                name = field if statistic == "instant" else field+"_"+statistic
                variable = ds.createVariable(name, "f4", ("Time", "point"))
                variable.units = FIELDS[field][2]
                variable.coordinates = "XLAT XLONG"
                variable.statistic = statistic
                variable.cell_methods = "time: " + {"instant": "point", "min": "minimum", "max": "maximum",
                                                     "mean": "mean", "accumulation": "sum"}[statistic]
                variable[:] = values[:, k]
        if journal is not None:
            journal.publish(temporary, target)
        else:
            from gpuwrf.io.wrfout_writer import _publish_wrfout_noreplace
            _publish_wrfout_noreplace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class AsyncStreamWriter:
    """One writer thread, at most two retained host chunks, fail-closed errors."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="regional-stream-writer")
        self.pending = []
        self.written = {}

    def submit(self, key, *args):
        if len(self.pending) >= 2:
            self._complete(self.pending.pop(0))
        self.pending.append((key, self.pool.submit(write_frames, *args)))

    def _complete(self, item):
        key, future = item
        self.written.setdefault(key, []).append(str(future.result()))

    def drain(self):
        while self.pending:
            self._complete(self.pending.pop(0))

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)
