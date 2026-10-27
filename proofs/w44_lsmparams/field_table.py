"""Per-field ON-default round-off table (manager ruling 2026-10-25 05:12Z; LSMPARAMS.md).

Every differing carry field gets: cells, median/max ULP, max abs, field max and max abs / field max. The field
max is taken over the Noah-MP land columns without land ice (State.xland == 1 and IVGTYP != ISICE of the case
wrfinput; |x| >= 1e30 fills excluded) whenever the field ends in the mass-grid (ny, nx) shape, else over all
cells. Packed history families (PackedFields, one [k, ny, nx] leaf per family) are split into named slices.
Raw numbers only: the bound is applied by the reader (CPU LS71 and the GPU prime use the same code).
Used in-process by root_gpu.py and post hoc by root_cause_table.py.
"""
from __future__ import annotations

import re

import numpy as np

FILL = 1e30


def ulps(x, y):
    kind = {4: np.int32, 8: np.int64}[x.dtype.itemsize]
    sign = (1 << (8 * x.dtype.itemsize - 1)) - 1

    def ordered(v):
        bits = v.view(kind).astype(np.int64) if kind is np.int32 else v.view(kind)
        return np.where(bits < 0, -(bits & sign), bits)
    return np.abs(ordered(x).astype(np.float64) - ordered(y).astype(np.float64))


def label(path, carry_slots, state_slots, land_slots):
    """keystr path below the domain key -> unique readable field name."""
    head = re.match(r"\[<flat index (\d+)>\]", path)
    if head is None:
        return path
    top, rest = carry_slots[int(head.group(1))], path[head.end():]
    inner = re.match(r"\[<flat index (\d+)>\]", rest)
    if top == "state" and inner:
        return "State." + state_slots[int(inner.group(1))] + rest[inner.end():]
    if top == "noahmp_land" and inner:
        return "noahmp_land." + land_slots[int(inner.group(1))] + rest[inner.end():]
    if inner and inner.end() == len(rest):
        return f"{top}[{inner.group(1)}]"
    return top + rest


def named_fields(host, domain, carry_slots, state_slots, land_slots, packed):
    """{keystr: array} of a domain-keyed carry tree -> {field name: array} of one domain."""
    prefix, out = f"['{domain}']", {}
    for key, value in host.items():
        if not key.startswith(prefix):
            continue
        name = label(key[len(prefix):], carry_slots, state_slots, land_slots)
        family = name[:-3] if name.endswith("[0]") else None
        if family in packed:
            assert value.shape[0] == len(packed[family]), (family, value.shape, len(packed[family]))
            for k, field in enumerate(packed[family]):
                out[f"{family}.{field}"] = value[k]
            continue
        assert name not in out, name
        out[name] = value
    return out


def scale_mask(fields, ice):
    land = np.asarray(fields["State.xland"]) == 1
    assert land.shape == ice.shape
    return land & ~ice


def row(name, x, y, mask):
    assert x.shape == y.shape and x.dtype == y.dtype, name
    m = x != y
    out = {"field": name, "dtype": str(x.dtype), "cells": int(m.sum())}
    if x.dtype.kind != "f":
        return {**out, "non_float": True, "max_abs_over_field_max": float("inf")}
    use = np.broadcast_to(mask, x.shape) if x.shape[-2:] == mask.shape else np.ones(x.shape, bool)
    finite = np.abs(x[use & np.isfinite(x) & (np.abs(x) < FILL)])
    scale = float(finite.max()) if finite.size else 0.0
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    u = ulps(x, y)
    worst = np.unravel_index(np.argmax(np.where(m, u, -1)), x.shape)
    return {**out, "ulp_median": float(np.median(u[m])), "ulp_max": int(u.max()), "max_abs": float(d.max()),
            "field_abs_max": scale, "scale_cells": "land_no_ice" if x.shape[-2:] == mask.shape else "all",
            "cells_outside_scale_columns": int((m & ~use).sum()),
            "max_abs_over_field_max": float(d.max() / scale) if scale else float("inf"),
            "value_at_max_ulp": float(x[worst])}


def table(reference, candidate, mask):
    assert reference.keys() == candidate.keys()
    rows = [row(name, reference[name], candidate[name], mask) for name in reference
            if reference[name].tobytes() != candidate[name].tobytes()]
    return sorted(rows, key=lambda r: -r["max_abs_over_field_max"])


def ice_mask(wrfinput):
    import netCDF4
    with netCDF4.Dataset(wrfinput) as ds:
        return np.asarray(ds["IVGTYP"][0]) == int(getattr(ds, "ISICE"))


def live_slots(carry):
    """Slot names and packed-family names straight from a live OperationalCarry (GPU prime)."""
    import dataclasses
    carry_slots = [f.name for f in dataclasses.fields(carry)]
    packed = {name: tuple(getattr(carry, name).names) for name in carry_slots
              if type(getattr(carry, name)).__name__ == "PackedFields"}
    return carry_slots, list(type(carry.state).__slots__), list(type(carry.noahmp_land).__slots__), packed


def domain_tables(reference_host, candidate_host, carries, input_dir):
    """In-process (root_gpu prime): one table per domain, labels checked against the live carry."""
    tables = {}
    for domain, carry in carries.items():
        slots = live_slots(carry)
        ref = named_fields(reference_host, domain, *slots)
        cand = named_fields(candidate_host, domain, *slots)
        # Labelling guard: named fields must be the live attributes of the reference carry.
        for name, value in (("State.xland", carry.state.xland), ("State.qfx", carry.state.qfx),
                            ("noahmp_land.smois", carry.noahmp_land.smois)):
            assert name in ref and ref[name].tobytes() == np.asarray(value).tobytes(), name
        rows = table(ref, cand, scale_mask(ref, ice_mask(input_dir / f"wrfinput_{domain}")))
        tables[domain] = {"rows": rows, "max_ratio": max((r["max_abs_over_field_max"] for r in rows), default=0.0),
                          "fields_compared": len(ref)}
    return tables
