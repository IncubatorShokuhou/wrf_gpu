"""CPU-only, mass-frozen WRF REAL lateral-boundary perturbation prototype.

Input is {WRF time: {U, V, T, QVAPOR: physical deviation arrays}} on the
full C grid. T is a DRY potential-temperature increment in K (also the dry
perturbation-T increment). Include every Times record AND the final nextbdytime
endpoint; missing fields mean zero. An optional MU array must be identically
zero. Static MUB/C1H/C2H/map factors come from the matching sibling wrfinput.

Source: pristine WRF main/real_em.F:866-878,1052-1064;
dyn_em/module_big_step_utilities_em.F:403-635; share/module_bc.F:2893-3103.
Scalar mass preserves WRF's split REAL grouping, not c1*(MU+MUB)+c2.
Stored U/V are field*hybrid_staggered_mass/MAPFAC_UY,VX; scalar T/QV
use hybrid mass without a map divisor. T stores moist theta perturbation
when USE_THETA_M=1. Only stored boundary arrays and their BT arrays change.
No dynamics, thermodynamic rebalance, ensemble calibration or forecast claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

_SIDES = ("XS", "XE", "YS", "YE")
_FIELDS = ("U", "V", "T", "QVAPOR")
_TIME = "%Y-%m-%d_%H:%M:%S"
_THIS = "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
_NEXT = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
_RV = float(np.float32(np.float32(461.6) / np.float32(287.)))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(value):
    if isinstance(value, str):
        return datetime.strptime(value, _TIME)
    if isinstance(value, datetime):
        if value.tzinfo:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        if not value.microsecond:
            return value
    raise ValueError("record timestamps must be exact-second UTC datetimes or WRF Times strings")


def _times(ds, name):
    return [_stamp(b"".join(np.asarray(row, dtype="S1").tolist()).decode().strip("\0 "))
            for row in ds[name][:]]


def _raw(ds, name, index=0):
    var = ds[name]
    if var.dtype != np.dtype("f4") or var.dimensions[:1] != ("Time",):
        raise ValueError(f"{name}: requires native WRF REAL(Time,...) storage")
    if any(key in var.ncattrs() for key in ("scale_factor", "add_offset", "least_significant_digit", "_QuantizeBitGroomNumberOfSignificantDigits")):
        raise ValueError(f"{name}: packed/quantized storage is unsupported")
    a = np.ma.asarray(var[index])
    if np.ma.getmaskarray(a).any() or not np.isfinite(np.asarray(a)).all():
        raise ValueError(f"{name}: missing/nonfinite source data")
    return np.asarray(a, dtype=np.float32)


def _strips(a, width):
    # All fields end in (y,x); boundary order is (width,z?,tangent).
    if a.ndim == 2:
        return {"XS": a[:, :width].T, "XE": a[:, -width:][:, ::-1].T,
                "YS": a[:width], "YE": a[-width:][::-1]}
    return {"XS": np.moveaxis(a[..., :width], -1, 0), "XE": np.moveaxis(a[..., -width:][..., ::-1], -1, 0),
            "YS": np.moveaxis(a[:, :width, :], 1, 0), "YE": np.moveaxis(a[:, -width:, :][:, ::-1, :], 1, 0)}


class _Context:
    def __init__(self, boundary, wrfinput):
        self.path = Path(wrfinput or Path(boundary).parent / "wrfinput_d01").resolve(strict=True)
        with Dataset(self.path) as ds:
            self.mub = _raw(ds, "MUB")
            self.c1, self.c2 = _raw(ds, "C1H"), _raw(ds, "C2H")
            self.mapu, self.mapv = _raw(ds, "MAPFAC_UY"), _raw(ds, "MAPFAC_VX")
            self.theta_m = int(ds.getncattr("USE_THETA_M"))
            self.attrs = {k: ds.getncattr(k) for k in ("GRID_ID", "DX", "DY", "HYBRID_OPT", "USE_THETA_M")}
        self.ny, self.nx = self.mub.shape
        self.nz = len(self.c1)
        if self.c1.ndim != 1 or self.c2.shape != self.c1.shape or self.theta_m not in (0, 1):
            raise ValueError("invalid hybrid coordinate or USE_THETA_M")
        if self.mapu.shape != (self.ny, self.nx + 1) or self.mapv.shape != (self.ny + 1, self.nx):
            raise ValueError("map-factor C-grid shape mismatch")
        if np.any(self.mapu <= 0) or np.any(self.mapv <= 0):
            raise ValueError("map factors must be positive")

    def admit(self, ds):
        for key, value in self.attrs.items():
            if key not in ds.ncattrs() or ds.getncattr(key) != value:
                raise ValueError(f"wrfinput/wrfbdy grid/convention mismatch: {key}")
        self.width = len(ds.dimensions["bdy_width"])
        if not 1 <= self.width <= min(self.nx, self.ny):
            raise ValueError("invalid boundary width")
        times, this, next_ = _times(ds, "Times"), _times(ds, _THIS), _times(ds, _NEXT)
        if not times or times != this or len(next_) != len(times):
            raise ValueError("boundary Times/thisbdytime metadata disagree")
        self.interval = (next_[0] - times[0]).total_seconds()
        if self.interval <= 0 or any(b - a != timedelta(seconds=self.interval) for a, b in zip(times, next_)):
            raise ValueError("invalid nextbdytime interval")
        if any(b - a != timedelta(seconds=self.interval) for a, b in zip(times, times[1:])):
            raise ValueError("Times must increase at the boundary interval")
        self.times = [*times, next_[-1]]
        self.nt = len(times)

    def read(self, ds, field, side, index):
        k = min(index, self.nt - 1)
        value = _raw(ds, f"{field}_B{side}", k)
        tendency = _raw(ds, f"{field}_BT{side}", k)
        if tendency.shape != value.shape:
            raise ValueError(f"{field}_{side}: base/tendency shapes differ")
        if index == self.nt:
            value = np.float32(value + np.float32(tendency * np.float32(self.interval)))
        return value

    def factors(self, ds, index):
        mu = np.zeros_like(self.mub)
        seen = np.zeros_like(mu, dtype=bool)
        w = self.width
        for side in _SIDES:
            strip = self.read(ds, "MU", side, index)
            target = _strips(mu, w)[side]
            known = _strips(seen, w)[side]
            if strip.shape != target.shape or not np.array_equal(target[known], strip[known]):
                raise ValueError("MU side shape/corner disagreement")
            target[...] = strip
            known[...] = True
        # Literal REAL grouping of calc_mu_staggered, including physical edges.
        u = np.empty((self.ny, self.nx + 1), np.float32)
        v = np.empty((self.ny + 1, self.nx), np.float32)
        u[:, 0] = mu[:, 0] + self.mub[:, 0]
        u[:, -1] = mu[:, -1] + self.mub[:, -1]
        u[:, 1:-1] = np.float32(.5) * (((mu[:, 1:] + mu[:, :-1]) + self.mub[:, 1:]) + self.mub[:, :-1])
        v[0] = mu[0] + self.mub[0]
        v[-1] = mu[-1] + self.mub[-1]
        v[1:-1] = np.float32(.5) * (((mu[1:] + mu[:-1]) + self.mub[1:]) + self.mub[:-1])
        c1, c2 = self.c1[:, None, None], self.c2[:, None, None]
        mass = (c1 * mu[None]) + (c1 * self.mub[None] + c2)
        # Host f64 division avoids an extra rounded mass/map factor. File values
        # are still REAL. The actual pristine scalar/staggered masses are REAL.
        result = {"T": _strips(mass, w), "QVAPOR": _strips(mass, w),
                  "U": _strips((c1 * u[None] + c2).astype(np.float64) / self.mapu[None], w),
                  "V": _strips((c1 * v[None] + c2).astype(np.float64) / self.mapv[None], w)}
        if any(np.any(a <= 0) or not np.isfinite(a).all() for sides in result.values() for a in sides.values()):
            raise ValueError("hybrid coupling mass must be finite and positive")
        return result

    def physical(self, coupled, factors):
        fields = {name: {s: coupled[name][s].astype(np.float64) / factors[name][s] for s in _SIDES} for name in _FIELDS}
        for side in _SIDES:
            full = fields["T"][side] + 300.
            if self.theta_m:
                full /= 1. + _RV * fields["QVAPOR"][side]
            fields["T"][side] = full  # Full DRY theta, K (deviation equals dry T' deviation).
        if any(not np.isfinite(a).all() or np.any(np.abs(a) > np.finfo(np.float32).max)
               for sides in fields.values() for a in sides.values()):
            raise ValueError("physical boundary state is not finite WRF REAL")
        return fields


def _records(context, deviations):
    if not isinstance(deviations, Mapping):
        raise ValueError("deviations_by_record must map exact timestamps to field dictionaries")
    normalized = {}
    shapes = {"U": (context.nz, context.ny, context.nx + 1), "V": (context.nz, context.ny + 1, context.nx),
              "T": (context.nz, context.ny, context.nx), "QVAPOR": (context.nz, context.ny, context.nx), "MU": (context.ny, context.nx)}
    for key, fields in deviations.items():
        stamp = _stamp(key)
        if stamp in normalized or not isinstance(fields, Mapping) or set(fields) - set(shapes):
            raise ValueError("duplicate timestamp/unknown field in deviations")
        normalized[stamp] = {}
        for name, value in fields.items():
            raw = np.ma.asarray(value)
            if raw.shape != shapes[name] or raw.dtype.kind not in "fiu" or np.ma.getmaskarray(raw).any() or not np.isfinite(raw).all():
                raise ValueError(f"{name}: deviations require finite unmasked physical C-grid arrays of shape {shapes[name]}")
            a = np.asarray(raw)
            if name == "MU" and np.any(a != 0):
                raise ValueError("MU is frozen; mass deviations are unsupported")
            normalized[stamp][name] = a
    if set(normalized) != set(context.times):
        raise ValueError("deviations must cover every Times record and the explicit final nextbdytime endpoint")
    return [normalized[t] for t in context.times]


def _target(context, ds, index, deviations):
    factors = context.factors(ds, index)
    old = {name: {s: context.read(ds, name, s, index) for s in _SIDES} for name in _FIELDS}
    physical = context.physical(old, factors)
    delta = {name: _strips(deviations[name], context.width) if name in deviations else {} for name in _FIELDS}
    result = {name: {} for name in _FIELDS}
    for side in _SIDES:
        dq = delta["QVAPOR"].get(side, 0.)
        qtarget = physical["QVAPOR"][side] + dq
        if np.any(qtarget < 0):
            raise ValueError("negative QVAPOR requested; no moisture clipping")
        encoded_q = np.float32(old["QVAPOR"][side].astype(np.float64) + dq * factors["QVAPOR"][side])
        result["QVAPOR"][side] = np.where(np.asarray(dq) == 0, old["QVAPOR"][side], encoded_q)
        dt = delta["T"].get(side, 0.)
        theta = physical["T"][side] + dt
        if np.any(theta <= 0):
            raise ValueError("nonpositive dry potential temperature requested")
        represented_q = result["QVAPOR"][side].astype(np.float64) / factors["QVAPOR"][side]
        thm = theta * (1. + _RV * represented_q) - 300. if context.theta_m else theta - 300.
        encoded = np.float32(thm * factors["T"][side])
        result["T"][side] = np.where((np.asarray(dt) == 0) & ((np.asarray(dq) == 0) | (not context.theta_m)), old["T"][side], encoded)
        for name in ("U", "V"):
            d = delta[name].get(side, 0.)
            encoded_wind = np.float32(old[name][side].astype(np.float64) + d * factors[name][side])
            result[name][side] = np.where(np.asarray(d) == 0, old[name][side], encoded_wind)
    if any(not np.isfinite(a).all() for sides in result.values() for a in sides.values()):
        raise ValueError("perturbation overflows WRF REAL storage")
    decoded = context.physical(result, factors)
    max_ulp = {}
    for name in _FIELDS:
        maximum = 0.
        for side in _SIDES:
            expected = delta[name].get(side, 0.)
            error = np.abs((decoded[name][side] - physical[name][side]) - expected)
            scale = np.maximum(np.abs(physical[name][side]), np.abs(physical[name][side] + expected)).astype(np.float32)
            ulp = np.abs(np.spacing(scale)).astype(np.float64)
            if not np.isfinite(error).all() or not np.isfinite(ulp).all() or np.any(ulp <= 0):
                raise ValueError(f"{name}: nonfinite physical decode gate")
            ratios = error / ulp
            maximum = max(maximum, float(np.max(ratios)))
            if np.any(error > ulp):
                raise ValueError(f"{name}: requested physical deviation exceeds one REAL-state ULP after encoding")
        max_ulp[name] = maximum
    return result, old, max_ulp


def perturb_wrfbdy(control_wrfbdy, deviations_by_record, out, *, wrfinput=None):
    """Atomically publish a new file; return hashes and physical decode-gate metrics.

    Zero deviations copy bytes without a NetCDF rewrite. Nonzero deviations are
    encoded with one final REAL storage rounding about the decoded control;
    host float64 arithmetic avoids cancellation of small increments. The decode
    gate is one REAL ULP at the physical STATE scale (full dry theta for T), not
    one ULP of a possibly tiny increment. Last BT needs an explicit endpoint.
    The implicit endpoint reconstructed from BT has its own tendency-rounding
    error; exact final interpolation/forecast qualification is not claimed.
    """
    source = Path(control_wrfbdy).resolve(strict=True)
    requested = Path(out).absolute()
    if requested.exists() or requested.is_symlink():
        raise ValueError("output must be a new file; refusing overwrite")
    output = requested.parent.resolve() / requested.name
    if source.parent == output.parent or source.parent in output.parents:
        raise ValueError("output must be outside the read-only source case")
    context = _Context(source, wrfinput)
    before, context_sha = _sha(source), _sha(context.path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".lbc-perturb-", dir=output.parent)
    os.close(fd)
    temporary = Path(path)
    changed = set()
    maxima = {name: 0. for name in _FIELDS}
    try:
        with Dataset(source) as src:
            context.admit(src)
            records = _records(context, deviations_by_record)
            zero = all(not np.any(a) for record in records for a in record.values())
            shutil.copyfile(source, temporary)
            current, old_current, metrics = _target(context, src, 0, records[0])
            with Dataset(temporary, "r+") if not zero else _NoWrite() as dst:
                for k in range(context.nt):
                    following, old_following, next_metrics = _target(context, src, k + 1, records[k + 1])
                    for name in _FIELDS:
                        maxima[name] = max(maxima[name], metrics[name], next_metrics[name])
                        for side in _SIDES:
                            base = f"{name}_B{side}"
                            tend = f"{name}_BT{side}"
                            a, b = current[name][side], following[name][side]
                            olda, oldb = old_current[name][side], old_following[name][side]
                            changed_a, changed_b = not np.array_equal(a.view("u4"), olda.view("u4")), not np.array_equal(b.view("u4"), oldb.view("u4"))
                            if changed_a:
                                dst[base][k] = a
                                changed.add(base)
                            if changed_a or changed_b:
                                value = np.float32(np.float32(b - a) / np.float32(context.interval))
                                if not np.isfinite(value).all():
                                    raise ValueError("nonfinite boundary tendency")
                                dst[tend][k] = value
                                changed.add(tend)
                    current, old_current, metrics = following, old_following, next_metrics
        if _sha(source) != before or _sha(context.path) != context_sha:
            raise ValueError("control/context changed during perturbation")
        digest = _sha(temporary)
        if zero and digest != before:
            raise ValueError("zero-deviation file identity failed")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, output)  # atomic no-clobber publication
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema": 1, "control": str(source), "control_sha256": before,
            "wrfinput": str(context.path), "wrfinput_sha256": context_sha,
            "output": str(output), "output_sha256": digest, "zero_deviation": zero,
            "records": context.nt, "endpoint": context.times[-1].strftime(_TIME),
            "interval_seconds": context.interval, "changed_variables": sorted(changed),
            "physical_encoding_max_state_ulp": maxima, "MU_policy": "frozen values and tendencies",
            "theta_policy": "dry increment -> moist THM using represented QV; full dry theta ULP",
            "terminal_policy": "explicit nextbdytime target; BT REAL(next-this)/interval; interpolation rounding separate",
            "scope": "CPU perturbation/storage prototype; no forecast qualification"}


class _NoWrite:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False
