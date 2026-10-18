"""Counted moisture policy and positivity / water-budget gates for LETKF analyses.

An ensemble update is linear in the state, so analysed water-vapour mixing ratios can go negative where
the prior spread is large compared with the background value (dry layers, inflated members). The probe
treated ANY negative cell as a binary refusal; one cell then flipped the decision between two otherwise
near-identical analyses (A1 STATUS R10). This module replaces that with a declared, counted policy:

``negative``
    ``"clip"`` (default): negative QVAPOR is set to 0 and the vapour this ADDS is integrated per
    member as dry-air-mass-weighted column water [kg m-2]. The analysis is admissible only while the
    added water stays inside the declared budget (``max_added_column_kg_m2`` per column and
    ``max_added_fraction`` of the member's total vapour).
    ``"refuse"``: no change; any negative cell makes the analysis inadmissible (old behaviour).
    ``"log"``: QVAPOR is analysed as z = ln(qv + floor) with the same LETKF weights and transformed
    back, qv = exp(z) - floor, so qv > -floor by construction; the residual (at most ``floor``) is
    clipped and counted with the same budget. This is a Gaussian-anamorphosis variant, opt-in.
``supersaturation``
    ``"keep"`` only (the model condenses analysed supersaturation in its first steps; the cells are counted).
    ``"cap"``/``"to_cloud"`` are refused in this build: applied after the boundary taper they would change the
    protected specified/relaxation zone, the fixed-QV rebalance does not load condensate changes, and the member
    writer does not carry QCLOUD.

Every policy reports the same counters; nothing is changed silently. The gates are engineering
limits declared here, not values fitted to any verification case; the receipt records the limits and
the measured values so a user can tighten or loosen them knowingly.

Dry-air mass per layer uses the WRF hybrid full-level dry pressure p_d = C3F*(MUB+MU) + C4F + P_TOP
(``geo`` from ``state_io.read_geometry``), m = (p_d(k) - p_d(k+1)) / g; geometries without the
vertical coordinate fall back to mass-level pressure differences (labelled ``approximate``).
Host NumPy only (fp64 arithmetic); nothing here touches the forecast program.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

G = 9.81                                     # WRF module_model_constants g
EP_2 = 287.0 / 461.6                         # R_d / R_v
SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15
RCP = 287.0 / 1004.5

NEGATIVE_POLICIES = ("clip", "refuse", "log")
SUPERSATURATION_POLICIES = ("keep",)
REFUSED_SUPERSATURATION = ("cap", "to_cloud")


@dataclass(frozen=True)
class MoisturePolicy:
    negative: str = "clip"
    supersaturation: str = "keep"
    log_floor_kgkg: float = 1.0e-7
    max_added_column_kg_m2: float = 0.1      # 0.1 mm precipitable water in any one column
    max_added_fraction: float = 1.0e-3       # 0.1 % of the member's domain vapour mass

    def __post_init__(self):
        if self.negative not in NEGATIVE_POLICIES:
            raise ValueError(f"moisture.negative must be one of {NEGATIVE_POLICIES}, got {self.negative!r}")
        if self.supersaturation in REFUSED_SUPERSATURATION:
            raise ValueError(f"moisture.supersaturation = {self.supersaturation!r} is refused in this build: it "
                             "would change the protected boundary zone and condensate that the fixed-QV rebalance "
                             "and the writers do not carry; use 'keep'")
        if self.supersaturation not in SUPERSATURATION_POLICIES:
            raise ValueError(f"moisture.supersaturation must be one of {SUPERSATURATION_POLICIES}, "
                             f"got {self.supersaturation!r}")
        for name in ("log_floor_kgkg", "max_added_column_kg_m2", "max_added_fraction"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and np.isfinite(value) and value >= 0):
                raise ValueError(f"moisture.{name} must be a finite number >= 0, got {value!r}")
        if self.negative == "log" and not self.log_floor_kgkg > 0:
            raise ValueError("moisture.log_floor_kgkg must be > 0 for the log policy")

    @classmethod
    def from_params(cls, params) -> "MoisturePolicy":
        """Policy from the LetkfParams policy names (default limits)."""
        return cls(negative=params.negative_qv, supersaturation=params.supersaturation)

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------- transforms

def to_log(qv, floor):
    return np.log(np.maximum(np.asarray(qv, np.float64), 0.0) + floor)


def from_log(z, floor):
    return np.exp(np.asarray(z, np.float64)) - floor


# ------------------------------------------------------------------------------------------- dry-air mass

def _col(geo, name):
    return np.asarray(geo[name], np.float64).reshape(-1)[:, None, None]


def layer_dry_mass(geo, mu=None, p_full=None):
    """Dry-air mass per layer [kg m-2] broadcast to the member axis, and a label.

    ``mu`` (N, ny, nx) perturbation dry column mass (wrfout MU); ``p_full`` (N, nz, ny, nx) the fallback.
    """
    if mu is not None and all(k in geo for k in ("C3F", "C4F", "P_TOP", "MUB")):
        mut = np.asarray(geo["MUB"], np.float64) + np.asarray(mu, np.float64)          # (N, ny, nx)
        pf = _col(geo, "C3F") * mut[:, None] + _col(geo, "C4F") + float(geo["P_TOP"])  # (N, nz+1, ny, nx)
        return (pf[:, :-1] - pf[:, 1:]) / G, "WRF hybrid dry pressure C3F*(MUB+MU)+C4F+P_TOP"
    if p_full is None:
        raise ValueError("layer_dry_mass needs MU with C3F/C4F/P_TOP/MUB or the full pressure")
    p = np.asarray(p_full, np.float64)
    return np.abs(np.gradient(p, axis=1)) / G, "approximate: |dp/dk| of mass-level full pressure"


def _area_weight(geo, shape2d):
    """Relative cell area 1/m^2 (map factor) for domain integrals; 1 where no map factor is given."""
    if "MAPFAC_M" in geo and np.shape(geo["MAPFAC_M"]) == tuple(shape2d):
        m = np.asarray(geo["MAPFAC_M"], np.float64)
        return 1.0 / (m * m)
    return np.ones(shape2d)


# ------------------------------------------------------------------------------------------- policy steps

def negative_step(xa, policy: MoisturePolicy, xb=None):
    """Apply the negative-value policy to one analysed moisture field (N, ...); return (field, raw, counts).

    ``raw`` is the field before any change (needed for the water budget). With the background ``xb`` only cells
    the analysis changed are treated: a negative value the background already had and the analysis left
    untouched (e.g. inside the tapered boundary zone) is counted as ``background_negative_cells`` and kept, so a
    zero increment stays bitwise the background."""
    raw = np.asarray(xa, np.float64)
    negative = raw < 0
    if xb is not None:
        inherited = negative & (raw == np.asarray(xb, np.float64))
        negative &= ~inherited
    counts = {"negative_cells": int(negative.sum()),
              "negative_min_kgkg": float(raw[negative].min()) if negative.any() else 0.0}
    if xb is not None:
        counts["background_negative_cells"] = int(inherited.sum())
    if policy.negative == "refuse":
        return raw, raw, counts
    out = np.where(negative, 0.0, raw)
    counts["clipped_cells"] = counts["negative_cells"]
    return out, raw, counts


def water_budget(raw, clipped, mass, geo):
    """Vapour ADDED by clipping (``clipped - raw`` >= 0) per member: column max and domain fraction."""
    added = (np.asarray(clipped, np.float64) - np.asarray(raw, np.float64)) * mass        # (N, nz, ny, nx)
    column = added.sum(axis=1)                                                            # (N, ny, nx)
    area = _area_weight(geo, column.shape[-2:])
    total = (np.asarray(clipped, np.float64) * mass).sum(axis=1)                          # (N, ny, nx)
    added_dom = (column * area).sum(axis=(-2, -1))
    total_dom = (total * area).sum(axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(total_dom > 0, added_dom / total_dom, np.where(added_dom > 0, np.inf, 0.0))
    return {"added_column_max_kg_m2": float(column.max()) if column.size else 0.0,
            "added_column_max_by_member_kg_m2": [float(v) for v in column.reshape(column.shape[0], -1).max(axis=1)],
            "added_fraction_max": float(fraction.max()) if fraction.size else 0.0,
            "added_fraction_by_member": [float(v) for v in fraction],
            "columns_with_added_water": int((column > 0).sum())}


def gate(policy: MoisturePolicy, counts: dict, budget: dict | None) -> tuple[bool, list[str]]:
    """Admissibility from the counted moisture result; returns (admissible, reasons)."""
    reasons = []
    if counts.get("nonfinite_cells", 0):
        reasons.append(f"non-finite analysed QVAPOR in {counts['nonfinite_cells']} cells")
    if policy.negative == "refuse":
        if counts.get("negative_cells", 0):
            reasons.append(f"negative analysed QVAPOR in {counts['negative_cells']} cells (policy 'refuse')")
        return not reasons, reasons
    if budget is not None:
        values = (budget["added_column_max_kg_m2"], budget["added_fraction_max"])
        if not all(np.isfinite(v) for v in values):          # NaN must never pass a '>' comparison silently
            reasons.append(f"water budget not finite {values}")
            return False, reasons
        if budget["added_column_max_kg_m2"] > policy.max_added_column_kg_m2:
            reasons.append(f"clipping added {budget['added_column_max_kg_m2']:.4g} kg m-2 in one column "
                           f"> limit {policy.max_added_column_kg_m2:.4g}")
        if budget["added_fraction_max"] > policy.max_added_fraction:
            reasons.append(f"clipping added {budget['added_fraction_max']:.4g} of a member's vapour "
                           f"> limit {policy.max_added_fraction:.4g}")
    return not reasons, reasons


# ------------------------------------------------------------------------------------------- saturation

def saturation_mixing_ratio(t_k, p_pa):
    """qsat over liquid water [kg/kg] (WRF svp1..svpt0 Bolton form, mixing ratio)."""
    t_k = np.asarray(t_k, np.float64)
    p_pa = np.asarray(p_pa, np.float64)
    es = 1000.0 * SVP1 * np.exp(SVP2 * (t_k - SVPT0) / (t_k - SVP3))
    return EP_2 * es / np.maximum(p_pa - es, 1.0)


def temperature(theta_pert, p_full):
    """Air temperature [K] from wrfout T (dry theta - 300 K) and full pressure [Pa]."""
    return (np.asarray(theta_pert, np.float64) + 300.0) * (np.asarray(p_full, np.float64) / 1.0e5) ** RCP


def supersaturation_counts(qv, theta_pert, p_full) -> dict:
    """Diagnostic only (policy 'keep'): analysed cells above saturation over liquid water; nothing is changed."""
    qv = np.asarray(qv, np.float64)
    qsat = saturation_mixing_ratio(temperature(theta_pert, p_full), p_full)
    excess = np.maximum(qv - qsat, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = np.where(qsat > 0, qv / qsat, np.nan)
    return {"policy": "keep", "supersaturated_cells": int((excess > 0).sum()),
            "excess_max_kgkg": float(excess.max()) if excess.size else 0.0,
            "rh_max": float(np.nanmax(rh)) if rh.size else 0.0}


def check_written(name, value, background=None) -> dict:
    """Positivity/finiteness census of a field exactly as it will be stored (after dtype cast).

    ``negative`` counts negative stored values except those inherited unchanged from ``background``."""
    arr = np.asarray(value)
    negative = arr < 0
    out = {"field": name, "dtype": str(arr.dtype), "nonfinite": int((~np.isfinite(arr)).sum()),
           "min": float(np.nanmin(arr)) if arr.size else 0.0}
    if background is not None:
        inherited = negative & (arr == np.asarray(background).astype(arr.dtype))
        out["inherited_negative"] = int(inherited.sum())
        negative &= ~inherited
    out["negative"] = int(negative.sum())
    return out
