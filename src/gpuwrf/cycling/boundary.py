"""Offline first-interval update, following pristine WRFDA da_update_bc.f90.

WRF 4.7.1 var/da/da_update_bc: MU :544-650; coupling :715-855;
edge extraction and endpoint-preserving tendencies :911-992. NumPy only.
"""
from datetime import datetime

import numpy as np

from .core import _dims, _read, _times

SIDES = ("XS", "XE", "YS", "YE")


def finite_field(ds, name, index=0):
    data, missing = _read(ds, name, index)
    var = ds[name]
    if missing.any() or data.dtype.kind != "f":
        raise ValueError(f"{name}: missing or non-floating values")
    if any(k in var.ncattrs() for k in (
            "scale_factor", "add_offset", "least_significant_digit",
            "_QuantizeBitGroomNumberOfSignificantDigits", "_QuantizeGranularBitRoundNumberOfSignificantDigits",
            "_QuantizeBitRoundNumberOfSignificantBits")):
        raise ValueError(f"{name}: packed/quantized fields are unsupported")
    return data


def coupled_field(ds, name):
    """WRFDA's REAL mass coupling, including staggered winds and moist THM."""
    field_name = "THM" if name == "T" and int(ds.USE_THETA_M) else name
    value = finite_field(ds, field_name)
    if name == "MU":
        return value
    mass = finite_field(ds, "MU") + finite_field(ds, "MUB")
    if name == "U":
        pad = np.pad(mass, ((0, 0), (1, 1)), mode="edge")
        mass = np.float32(.5) * (pad[:, 1:] + pad[:, :-1])
    elif name == "V":
        pad = np.pad(mass, ((1, 1), (0, 0)), mode="edge")
        mass = np.float32(.5) * (pad[1:] + pad[:-1])
    level = "F" if _dims(ds[field_name])[0] == "bottom_top_stag" else "H"
    c1, c2 = (finite_field(ds, f"C{n}{level}")[:, None, None] for n in (1, 2))
    value = value * (c1 * mass + c2)
    if name in ("U", "V", "W"):
        factor = finite_field(ds, "MAPFAC_" + (name if name != "W" else "M"))
        if np.any(factor <= 0):
            raise ValueError(f"{name}: non-positive map factor")
        value = value / factor
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite coupled field")
    return value


def edge(value, side, width):
    """NetCDF bdy order is (width, [z,] along-edge), reversed east/north."""
    axis = -1 if side[0] == "X" else -2
    if width > value.shape[axis]:
        raise ValueError("boundary width exceeds grid")
    indices = np.arange(width)
    if side[1] == "E":
        indices = value.shape[axis] - 1 - indices
    return np.moveaxis(np.take(value, indices, axis=axis), axis, 0)


def _clock(ds, name):
    # WRF metadata uses this encoded variable name; plain names are accepted
    # for direct NetCDF producers. WRFDA da_get_bdytimestr_cdf.inc constructs
    # the same prefix/suffix and reads the first 19 characters.
    encoded = "md___" + name + "e_x_t_d_o_m_a_i_n_m_e_t_a_data_"
    candidates = [n for n in (name, encoded) if n in ds.variables]
    if not candidates:
        raise ValueError(f"boundary requires {name}(Time,DateStrLen)")
    clocks = []
    for varname in candidates:
        var = ds[varname]
        if var.ndim != 2 or var.dimensions[0] != "Time" or var.shape[1] < 19:
            raise ValueError(f"boundary has invalid {varname} dimensions")
        clocks.append([np.asarray(row, dtype="S1").tobytes().decode("ascii")[:19] for row in var[:]])
    if any(clock != clocks[0] for clock in clocks[1:]):
        raise ValueError(f"boundary has conflicting {name} aliases")
    return clocks[0]


def update_first_interval(initial, boundary, *, allowed_fields):
    """Modify only record zero of an already private boundary copy.

    With >=2 records WRFDA retains the second VALUE (not the independently
    rounded extrapolated first value). With one record it reconstructs that
    endpoint from the original value + tendency*interval. Later bytes stay put.
    """
    stamps = _times(boundary)
    if not stamps or stamps[0] != _times(initial)[0]:
        raise ValueError("boundary first Times must equal the cycled initialization time")
    this, next_ = _clock(boundary, "thisbdytime"), _clock(boundary, "nextbdytime")
    if this != stamps or len(next_) != len(stamps):
        raise ValueError("boundary time metadata disagrees with Times")
    parse = lambda s: datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")
    intervals = [(parse(b) - parse(a)).total_seconds() for a, b in zip(this, next_)]
    if any(dt <= 0 or dt != intervals[0] for dt in intervals):
        raise ValueError("boundary intervals must be positive and constant")
    if any(next_[i] != stamps[i + 1] for i in range(len(stamps) - 1)):
        raise ValueError("boundary nextbdytime disagrees with following record")
    dt = intervals[0]
    for attr in ("GRID_ID", "DX", "DY", "MAP_PROJ", "HYBRID_OPT", "USE_THETA_M"):
        if attr not in boundary.ncattrs() or boundary.getncattr(attr) != initial.getncattr(attr):
            raise ValueError(f"boundary grid/coordinate metadata differs: {attr}")
    for dim in ("west_east", "south_north", "bottom_top", "bottom_top_stag",
                "west_east_stag", "south_north_stag"):
        if dim not in boundary.dimensions or len(boundary.dimensions[dim]) != len(initial.dimensions[dim]):
            raise ValueError(f"boundary grid dimension differs: {dim}")
    fields = set()
    unchanged = set()
    for name in boundary.variables:
        if "_B" in name:
            field, suffix = name.rsplit("_B", 1)
            if suffix in (*SIDES, *("T" + side for side in SIDES)):
                # These fresh boundary slots are outside the atmospheric
                # transfer and WRFDA's update list. HT_SHAD belongs to fresh
                # terrain; PC is the Registry hybrid-coordinate legacy slot
                # (the prognostic dry column in this port/WRFDA is MU).
                if field in ("HT_SHAD", "PC"):
                    unchanged.add(field)
                    finite_field(boundary, name)
                    continue
                if field not in allowed_fields:
                    raise ValueError(f"boundary field {field} has no cycled atmospheric source")
                fields.add(field)
    required = {"U", "V", "T", "PH", "MU", "QVAPOR"}
    if not required <= fields:
        raise ValueError(f"missing required boundary fields: {sorted(required - fields)}")
    report = {}
    for field in sorted(fields):
        full = coupled_field(initial, field)
        for side in SIDES:
            value_name, tendency_name = f"{field}_B{side}", f"{field}_BT{side}"
            value = finite_field(boundary, value_name)
            tendency = finite_field(boundary, tendency_name)
            var, ten = boundary[value_name], boundary[tendency_name]
            dims = ("Time", "bdy_width")
            if full.ndim == 3:
                dims += (_dims(initial["THM" if field == "T" and int(initial.USE_THETA_M) else field])[0],)
            dims += (("south_north_stag" if field == "V" else "south_north") if side[0] == "X"
                     else ("west_east_stag" if field == "U" else "west_east"),)
            if var.dimensions != dims or ten.dimensions != dims or var.dtype != ten.dtype:
                raise ValueError(f"{value_name}: incompatible boundary dimensions/dtype")
            start = edge(full, side, len(boundary.dimensions["bdy_width"]))
            if start.shape != value.shape or start.dtype != value.dtype:
                raise ValueError(f"{value_name}: incompatible coupled field shape/dtype")
            endpoint = (finite_field(boundary, value_name, 1) if len(stamps) > 1
                        else value + tendency * np.asarray(dt, dtype=value.dtype))
            # Retain exact original bytes for a no-change interval.
            new_tendency = ((endpoint - start) / np.asarray(dt, dtype=value.dtype)
                            if not np.array_equal(start, value) else tendency)
            if not np.isfinite(new_tendency).all():
                raise ValueError(f"{tendency_name}: non-finite recomputed tendency")
            error = np.abs(start.astype(np.float64) + new_tendency.astype(np.float64) * dt - endpoint)
            u = np.finfo(value.dtype).eps / 2
            bound = 4 * u / (1 - 4 * u) * (np.abs(start.astype(np.float64)) + np.abs(endpoint.astype(np.float64)))
            if float(np.asarray(dt, dtype=value.dtype)) != dt:
                raise ValueError("boundary interval must be exactly representable in its storage dtype")
            # Relative gamma bounds alone exclude subnormal REAL tendencies.
            # For s=fl(end-start), t=fl(s/dt), gradual underflow contributes
            # eta/2 from subtraction and dt*eta/2 from division. The extra
            # eta/2 covers the (1+u) factor; normal-rounding stays gamma_4.
            eta = float(np.nextafter(np.zeros((), dtype=value.dtype), np.ones((), dtype=value.dtype)))
            underflow_bound = eta * (dt + 2.) / 2.
            bound = bound + underflow_bound
            if np.any(error > bound):
                raise ValueError(f"{value_name}: first interval does not reach retained endpoint")
            var[0], ten[0] = start, new_tendency
            report[value_name] = {"start_sha256": _sha(start), "endpoint_sha256": _sha(endpoint),
                                  "endpoint_max_abs_error": float(error.max()),
                                  "endpoint_underflow_bound": underflow_bound,
                                  "endpoint_source": "second value" if len(stamps) > 1 else "original extrapolation"}
    return {"interval_seconds": dt, "updated_records": [0], "fields": report,
            "unchanged_fields": sorted(unchanged)}


def _sha(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
