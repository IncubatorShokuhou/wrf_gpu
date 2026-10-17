"""NumPy WRF pressure-coordinate mass/water budgets with shared diagnostic units.

Only storage changes are reported: without resolved lateral and surface flux
integrals these are not closed-domain conservation residuals or fidelity gates.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from .case import field
from .common import char_times, finite_number
from gpuwrf.io.budget_definitions import GRAVITY_M_S2, WATER_SPECIES


def offline_budget(paths):
    rows = []
    reference_area = None
    reference_layers = None
    reference_species = None
    reference_precip = None
    for path in paths:
        with Dataset(path) as ds:
            times = char_times(ds)
            for record, stamp in enumerate(times):
                f = lambda key: field(ds, key, record)
                dx = finite_number(getattr(ds, "DX", None), "DX", positive=True)
                dy = finite_number(getattr(ds, "DY", None), "DY", positive=True)
                mx, my = (f("MAPFAC_MX"), f("MAPFAC_MY")) if "MAPFAC_MX" in ds.variables and "MAPFAC_MY" in ds.variables else (f("MAPFAC_M"), f("MAPFAC_M"))
                if (mx <= 0).any() or (my <= 0).any():
                    raise ValueError("map factors must be positive")
                area = dx * dy / (mx * my)
                mu = f("MU") + f("MUB")
                if (mu <= 0).any():
                    raise ValueError("total dry column mass pressure must be positive")
                if "C3F" in ds.variables and "C4F" in ds.variables:
                    c3, c4 = f("C3F").ravel(), f("C4F").ravel()
                elif not {"C3F", "C4F"} & ds.variables.keys() and int(getattr(ds, "HYBRID_OPT", -1)) == 0:
                    c3 = f("ZNW").ravel()
                    c4 = np.zeros_like(c3)
                else:
                    raise ValueError("hybrid pressure requires C3F/C4F (sigma ZNW allowed only with HYBRID_OPT=0)")
                nz = len(ds.dimensions["bottom_top"])
                if c3.shape != (nz + 1,) or c4.shape != c3.shape:
                    raise ValueError("pressure coefficients must have one value per vertical face")
                # Pristine module_initialize_real.F:6742: p_dry=C3F*MU0+C4F+ptop.
                # Top pressure cancels. C1F/C2F are derivatives, not pressure.
                # Difference 1-D coefficients first to avoid full interface arrays.
                dp = (c3[:-1] - c3[1:])[:, None, None] * mu + (c4[:-1] - c4[1:])[:, None, None]
                if not np.isfinite(dp).all() or (dp <= 0).any():
                    raise ValueError("dry pressure layer thickness must be finite and positive")
                layer_mass = dp * area / GRAVITY_M_S2
                species = [name for name in WATER_SPECIES if name in ds.variables]
                missing = [name for name in WATER_SPECIES if name not in ds.variables]
                if "QVAPOR" not in species:
                    raise ValueError("QVAPOR required for moisture storage")
                water = 0.0
                for name in species:
                    q = f(name)
                    if q.shape != layer_mass.shape:
                        raise ValueError(f"{name} shape differs from hybrid layer mass")
                    water += float(np.sum(q * layer_mass, dtype=np.float64))
                precip = 0.0
                precip_names = ("RAINC", "RAINNC") + (("RAINSH",) if "RAINSH" in ds.variables else ())
                for name in precip_names:
                    rain = f(name)
                    bucket = "I_" + name
                    if bucket in ds.variables and np.any(f(bucket) != 0):
                        amount = finite_number(getattr(ds, "BUCKET_MM", None), "BUCKET_MM", positive=True)
                        rain = rain + f(bucket) * amount
                    precip += float(np.sum(rain * area, dtype=np.float64))
                if reference_area is not None and (not np.array_equal(area, reference_area) or
                        not np.array_equal(c3, reference_layers[0]) or not np.array_equal(c4, reference_layers[1]) or
                        species != reference_species or precip_names != reference_precip):
                    raise ValueError("budget records must share geometry, vertical coefficients and moisture fields")
                reference_area, reference_layers, reference_species = area, (c3, c4), species
                reference_precip = precip_names
                rows.append({"path": str(Path(path)), "time": stamp.isoformat(),
                             "dry_mass_kg": float(np.sum(mu * area / GRAVITY_M_S2, dtype=np.float64)),
                             "air_water_kg": water, "precip_accumulated_kg": precip,
                             "water_plus_precip_kg": water + precip,
                             "included_species": species, "missing_species": missing,
                             "included_precip_accumulators": list(precip_names),
                             "missing_precip_accumulators": [] if "RAINSH" in precip_names else ["RAINSH"]})
    rows.sort(key=lambda row: row["time"])
    if len(rows) < 2 or any(b["time"] <= a["time"] for a,b in zip(rows,rows[1:])):
        raise ValueError("budget needs at least two records with unique increasing times from one domain")
    deltas = {key: rows[-1][key] - rows[0][key] for key in ("dry_mass_kg", "air_water_kg", "precip_accumulated_kg", "water_plus_precip_kg")}
    if not all(np.isfinite(v) for row in rows for v in (row["dry_mass_kg"], row["air_water_kg"], row["precip_accumulated_kg"], row["water_plus_precip_kg"])):
        raise ValueError("budget overflow/nonfinite result")
    return {"schema": "GpuwrfOfflineBudget", "schema_version": 1, "records": rows, "changes": deltas,
            "closure_status": "UNKNOWN: lateral fluxes and integrated surface evaporation not supplied",
            "definitions": "pristine WRF C3F/C4F pressure-layer mass; shared conservation units; RAINC+RAINNC+available RAINSH includes all precip phases (no double-counted snow)",
            "sampling": "each supplied history record, float64 NumPy reductions; negative values retained"}
