"""Perturbed-LBC ensemble prototype — OPT-IN (lane w41-da-ens; R3 item 6, PLAN2 W3 boundary pilot).

R3 (2): with one shared wrfbdy (and the IC taper) the outer 15 rows (57 % of the ALISIOS d01) carry 0.36–0.90 of the
interior spread and the specified zone none, and the trade-wind inflow brings that unperturbed air into the interior
within 6–12 h [I]. This module gives every member its OWN wrfbdy whose boundary deviations

* at the first record equal the member's IC deviation (consistency at t0: the IC is then written UNTAPERED for
  U/V/T/QVAPOR; MU stays tapered because ``lbc_perturb`` keeps MU frozen in the boundary file), and
* evolve in time with the member's perturbation structure (e.g. the same Gaussian combination weights applied to
  time-lagged pool states valid at each record time — spatially and temporally coherent deviations),

encoded by ``lbc_perturb.perturb_wrfbdy`` (REAL storage, one-ULP decode gate, BT tendencies recomputed from the
perturbed endpoints, MU values/tendencies frozen). Records after ``perturb_until`` get zero deviation (bytes kept).
No forecast qualification is claimed by the encoding itself; the OSSE (proofs/a1_da/ens) measures the effect.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

LBC_FIELDS = ("U", "V", "T", "QVAPOR")
_TIME = "%Y-%m-%d_%H:%M:%S"


def record_times(wrfbdy) -> list[str]:
    """Every Times record plus the final nextbdytime endpoint (the keys ``lbc_perturb`` requires)."""
    from .lbc_perturb import _NEXT, _times
    with Dataset(str(wrfbdy)) as ds:
        times = _times(ds, "Times")
        nxt = _times(ds, _NEXT)
    return [t.strftime(_TIME) for t in [*times, nxt[-1]]]


def member_deviations(control_by_time: Mapping[str, Mapping[str, np.ndarray]],
                      member_by_time: Mapping[str, Mapping[str, np.ndarray]], times: Sequence[str],
                      *, perturb_until: str | None = None) -> dict:
    """{time: {U, V, T, QVAPOR: member − control}} for one member over ``times`` (zero dict after
    ``perturb_until``; a time missing from ``member_by_time`` is refused unless after ``perturb_until``)."""
    stop = datetime.strptime(perturb_until, _TIME) if perturb_until else None
    out = {}
    for t in times:
        if stop is not None and datetime.strptime(t, _TIME) > stop:
            out[t] = {}
            continue
        if t not in member_by_time or t not in control_by_time:
            raise ValueError(f"no member/control state at boundary record {t}")
        out[t] = {n: np.asarray(member_by_time[t][n], np.float64) - np.asarray(control_by_time[t][n], np.float64)
                  for n in LBC_FIELDS if n in member_by_time[t]}
    return out


def ic_consistency(ic_deviation: Mapping[str, np.ndarray], record0: Mapping[str, np.ndarray], width: int) -> dict:
    """max |IC deviation − first-record deviation| inside the ``width`` boundary rows, per field (must be 0)."""
    out = {}
    for name, dev in record0.items():
        a = np.asarray(ic_deviation[name], np.float64)
        b = np.asarray(dev, np.float64)
        ny, nx = a.shape[-2:]
        j, i = np.mgrid[0:ny, 0:nx]
        ring = np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i)) < width
        out[name] = float(np.abs(a - b)[..., ring].max()) if ring.any() else 0.0
    return out


def write_member_wrfbdys(control_wrfbdy, wrfinput, member_records: Mapping[str, Mapping], out_dir) -> dict:
    """One perturbed wrfbdy per member (``member_records``: label → {time: {field: deviation}}). Returns and writes
    ``LBC_MANIFEST.json`` with every encoding receipt (hashes, max state ULP, changed variables)."""
    from .lbc_perturb import perturb_wrfbdy

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"{out_dir}: new output directory required")
    out_dir.mkdir(parents=True)
    receipts = {}
    for label, records in member_records.items():
        receipts[label] = perturb_wrfbdy(control_wrfbdy, records, out_dir / label / "wrfbdy_d01", wrfinput=wrfinput)
    manifest = {"schema": "w41.lbc_ens.v1", "control_wrfbdy": str(control_wrfbdy), "wrfinput": str(wrfinput),
                "members": receipts}
    (out_dir / "LBC_MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest
