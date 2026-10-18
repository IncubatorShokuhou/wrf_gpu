"""Station observation operator for EKI: member wrfouts + observations -> (members x obs) matrices.

Reuses the merged station-verification operator (``gpuwrf.verify``: bilinear/land-aware sampling, earth-relative
winds, optional lapse-rate and log-wind height corrections) for every member with ONE set of extraction options,
then keeps only the (station, valid time, variable) keys that every member and the observations share, in a
fixed sorted order.

ALISIOS data (observations, station metadata, member wrfouts — real paths count, so symlinks into the ALISIOS
roots are protected too) are read only inside ONE ``alisios_guard`` session that admits the complete protected
read set and stays open until the last file is read (scan + drop-DENY + admit + read spy, ALLOW days only).
Without a guard such a read set is refused before anything is opened; purely local inputs need no guard.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

ALISIOS_DATA_ROOTS = ("<DATA_ROOT>/server/work/src/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/alisios")


def is_alisios_data(path) -> bool:
    p = Path(os.fsdecode(path)).resolve()
    return any(p == Path(r) or Path(r) in p.parents for r in ALISIOS_DATA_ROOTS)


def protected_paths(paths) -> list[str]:
    """Resolved paths of every protected input reachable from ``paths``: the path itself, and below a directory every
    link (directory aliases followed once, cycle-safe) -- the da-cycle walker (assimilation.letkf.member_driver), so
    the collector and the member driver share one implementation."""
    from gpuwrf.assimilation.letkf.member_driver import _walk

    found = []

    def check(path) -> bool:
        if is_alisios_data(path):
            found.append(str(Path(path).resolve()))
            return True
        return False

    for raw in paths:
        _walk(Path(os.fsdecode(raw)), check, namelists=False)
    return sorted(set(found))


@contextlib.contextmanager
def _read_session(read_set, guard):
    """Guard session over the protected part of ``read_set`` (refused without a guard); no-op for local inputs."""
    protected = protected_paths(read_set)
    if not protected:
        yield None
        return
    if guard is None:
        raise ValueError(f"{len(protected)} input(s) resolve into ALISIOS data roots (e.g. {protected[0]}); they are read "
                         "only through alisios_guard (pass issue_days/valid_days/receipt_dir)")
    try:
        from gpuwrf.assimilation.letkf.alisios_guard import guarded_session
    except ImportError as exc:  # never fall back to an unguarded read
        raise ValueError("gpuwrf.assimilation.letkf.alisios_guard is not importable; refusing the ALISIOS read") from exc
    allowed = {path: list(guard["valid_days"]) for path in protected}
    with guarded_session(allowed, guard["issue_days"], guard["receipt_dir"],
                         purpose="Track P EKI station predictions (calibration)") as session:
        yield session
    guard["receipt"] = session.receipt


def collect(member_wrfouts, stations_path, obs_path, *, variables=("t2", "wind"), sigma=None,
            obs_format="csv", guard=None, **extraction):
    """-> (predictions (N, m), y (m,), sigma (m,), keys [(station, iso time, variable)])."""
    from gpuwrf.verify import observations as obs_io
    from gpuwrf.verify import wrfout as wrfout_io

    if len(member_wrfouts) < 2:
        raise ValueError("need at least two members")
    sigma = dict(sigma or {})
    missing = [v for v in variables if v not in sigma]
    if missing:
        raise ValueError(f"observation error sigma required for {missing} (representation + instrument)")
    members = [[Path(p) for p in paths] for paths in member_wrfouts]
    read_set = [stations_path, obs_path, *(p for paths in members for p in paths)]

    def table(frame):
        frame = frame[frame.variable.isin(variables)]
        return {(r.station_uid, str(r.valid_time), r.variable): float(r.value) for r in frame.itertuples()}

    with _read_session(read_set, guard):  # one session for the whole read lifetime
        stations = obs_io.read_stations(stations_path)
        obs, _audit = obs_io.read_observations(obs_path, obs_format)
        obs_map = table(obs[obs.variable.isin(variables)])
        member_maps = []
        for paths in members:
            rows, _ = wrfout_io.extract_wrfout(paths, stations, **extraction)
            member_maps.append(table(rows))
    keys = sorted(set(obs_map).intersection(*member_maps))
    if not keys:
        raise ValueError("no common station/time/variable support between observations and every member")
    pred = np.array([[m[k] for k in keys] for m in member_maps], dtype=np.float64)
    y = np.array([obs_map[k] for k in keys], dtype=np.float64)
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite predictions or observations")
    sig = np.array([float(sigma[k[2]]) for k in keys], dtype=np.float64)
    if np.any(sig <= 0):
        raise ValueError("sigma must be positive")
    return pred, y, sig, keys
