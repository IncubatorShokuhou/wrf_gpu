"""Read native streams with exact-length, shape and finite-value checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_array(f, shape, dtype=np.float32):
    count = int(np.prod(shape, dtype=np.int64))
    a = np.fromfile(f, dtype, count)
    if a.size != count:
        raise ValueError(f"Truncated oracle: need {count}, got {a.size}")
    a = a.reshape(shape, order="F")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite oracle")
    return a


def tables(path):
    out = {}
    with Path(path).open("rb") as f:
        while name := f.read(64):
            if len(name) != 64:
                raise ValueError("Truncated table name")
            key = name.decode("ascii").strip()
            hdr = read_array(f, (4,), np.int32)
            rank, dims = int(hdr[0]), hdr[1:]
            if not 0 <= rank <= 3 or np.any(dims <= 0) or np.prod(dims) > 1000000 or key in out:
                raise ValueError("Invalid table header")
            out[key] = read_array(f, tuple(dims[:rank]))
    if not out:
        raise ValueError("Empty table stream")
    return out


def leaves(path):
    with Path(path).open("rb") as f:
        nc, nz = read_array(f, (2,), np.int32)
        if not 1 <= nc <= 10000 or not 5 <= nz <= 100:
            raise ValueError("Invalid column dimensions")
        shapes = {
            "dz": (nc, nz), "trp": (nc, nz), "ozone": (nc, nz + 1, 5),
            "continuum": (nc, nz, 18), "gas": (nc, nz, 121), "weights": (121,),
            "rain": (nc, nz, 6, 18), "graupel": (nc, nz, 6, 18), "rayleigh": (nc, nz, 6, 18),
        }
        out = {key: read_array(f, shape) for key, shape in shapes.items()}
        if f.read(1):
            raise ValueError("Trailing oracle bytes")
    return out


def operands(path):
    with Path(path).open("rb") as f:
        nc, nz = read_array(f, (2,), np.int32)
        scalars = read_array(f, (4,))
        year = read_array(f, (), np.int32)
        out = {"scalars": scalars, "year": year}
        for key, shape, dtype in (
            ("znw", (nz + 1,), np.float32), ("znu", (nz + 1,), np.float32),
            ("julian", (nc,), np.float32), ("tau", (nc,), np.float32),
            ("declin", (nc,), np.float32), ("solcon", (nc,), np.float32), ("julday", (nc,), np.int32),
            ("sfc", (nc, 11), np.float32), ("fields", (nc, nz + 1, 15), np.float32),
        ):
            out[key] = read_array(f, shape, dtype)
        if f.read(1):
            raise ValueError("Trailing operand bytes")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--input", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t, l = tables(args.build / "tables.bin"), leaves(args.build / "leaves.bin")
    np.savez_compressed(args.out / "flg_tables_v1.npz", **t)
    np.savez_compressed(args.out / "flg_leaves_v1.npz", **l, **{"in_" + k: v for k, v in operands(args.input).items()})
    print(json.dumps({"tables": len(t), "leaf_arrays": {k: list(v.shape) for k, v in l.items()}}))
