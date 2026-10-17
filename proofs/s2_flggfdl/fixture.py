"""Prepare explicit RAD_FLG/ETARA operands from retained real CPU-WRF columns.

The WRF history fixture did not retain T8W, ZNW, VEGFRA or cumulus accumulators.
Those are declared constructed inputs below; this is a routine-boundary oracle,
not a replay of the original CPU forecast's radiation call. No CAM output is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(source: Path, out: Path) -> None:
    with np.load(source, allow_pickle=False) as data:
        a = {key[3:]: data[key].copy() for key in data.files if key.startswith("in_")}
        a["declin"] = data["declin"].copy()
        a["solcon"] = data["solcon"].copy()
        categories = data["columns_category"].tolist()
    nc, nz = a["t"].shape
    # Explicit interface-temperature interpolation; a port must consume these
    # same operands. End interfaces use the nearest mass temperature.
    t8w = np.empty((nc, nz + 1), np.float32)
    t8w[:, 0], t8w[:, -1] = a["t"][:, 0], a["t"][:, -1]
    w = (a["p8w"][:, 1:-1] - a["p"][:, 1:]) / (a["p"][:, :-1] - a["p"][:, 1:])
    t8w[:, 1:-1] = a["t"][:, 1:] + w * (a["t"][:, :-1] - a["t"][:, 1:])
    # Construct a legal eta grid, then its exact arithmetic midpoint ZNU.
    # Retained ZNU cannot uniquely recover the original rounded ZNW.
    znw = np.linspace(1, 0, nz + 1, dtype=np.float32)
    znu = np.r_[np.float32(.5) * (znw[:-1] + znw[1:]), np.float32(0)]
    sfc = np.stack([a[k] for k in ("tsk", "emiss", "albedo", "xland", "xlat", "xlong", "snow")]
                   + [np.full(nc, 50., np.float32), np.zeros(nc, np.float32),
                      np.full(nc, nz + 1, np.float32), np.zeros(nc, np.float32)], axis=-1)
    names = ("t", "p", "pi", "rho", "t8w", "qv", "qc", "qr", "qi", "qs", "qg", "cldfra", "p8w", "dz8w", "o3")
    fields = np.zeros((nc, nz + 1, len(names)), np.float32)
    for n, key in enumerate(names):
        values = t8w if key == "t8w" else a.get(key)
        if values is not None:
            fields[:, :values.shape[1], n] = values
            if values.shape[1] == nz:
                fields[:, -1, n] = values[:, -1]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "input.bin").open("wb") as f:
        np.asarray([nc, nz], np.int32).tofile(f)
        np.asarray([a[k] for k in ("p_top", "gmt", "radt", "dt")], np.float32).tofile(f)
        np.asarray(a["yr"], np.int32).tofile(f)
        for arr in (znw, znu, a["julian"], a["tau"], a["declin"], a["solcon"], a["julday"], sfc, fields):
            arr.ravel(order="F").tofile(f)
    metadata = {
        "schema": "wrf-flg-gfdl-routine-operands-v1", "ncol": nc, "nz": nz,
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((out / "input.bin").read_bytes()).hexdigest(),
        "categories": categories, "tau_hours": a["tau"].tolist(),
        "species_nonzero_levels": {k: int(np.count_nonzero(a[k])) for k in ("qv", "qc", "qr", "qi", "qs", "qg")},
        "constructed": ["T8W pressure interpolation with nearest endpoint T", "linear ZNW and midpoint ZNU",
                        "VEGFRA=50 percent", "HTOP=0 HBOT=nz+1 CUPPT=0", "ozone input=0 (FLG prescribed NOZONE=1)"],
        "scope": "routine-boundary source oracle only; not original forecast-call replay",
        "gate_preregistered": "finite all outputs; candidate vs WRF REAL: 16 ulp + 1e-11 K/s tendencies, 16 ulp + 3e-6 W/m2 fluxes",
    }
    (out / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.out)
