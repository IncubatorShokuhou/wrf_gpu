"""Generate spectral tendency fixtures using original CPU-WRF Swiss fields.

Usage: python make_savepoints.py BUILD OUTPUT
No gpuwrf/JAX import: the reference is the unmodified WRF/FFTPACK executable.
Analysis pairs are other CPU-WRF hours, not a port-generated target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import netCDF4
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
FIELDS = ("u", "v", "t", "ph", "q")
SUFFIXES = ("uv", "t", "ph", "q")


def original_frame(hour, ny, nx):
    path = CPU / f"wrfout_d01_2023-01-15_{hour:02d}:00:00"
    with netCDF4.Dataset(path) as ds:
        a = lambda n: np.asarray(ds[n][0], np.float32)
        ys, xs = slice(14, 14 + ny), slice(16, 16 + nx)
        q = a("QVAPOR")[:, ys, xs]
        rvrd = np.float32(np.float32(461.6) / np.float32(287.0))
        th = (a("THM")[:, ys, xs] + np.float32(300)) / (np.float32(1) + rvrd * q)
        ph = a("PH")[:, ys, xs]
        return dict(u=a("U")[:, ys, 16:17 + nx], v=a("V")[:, 14:15 + ny, xs],
                    t=th - np.float32(300), ph=ph, q=q, th_phy=th,
                    ht=a("HGT")[ys, xs], pblh=a("PBLH")[ys, xs],
                    z_at_w=(a("PHB")[:, ys, xs] + ph) / np.float32(9.81),
                    mu_2=a("MU")[ys, xs], mub=a("MUB")[ys, xs],
                    c1h=a("C1H"), c2h=a("C2H"), c1f=a("C1F"), c2f=a("C2F")), path


def rectangular(a, nz, ny, nx, dtype):
    shape = (nz + 1, ny + 1, nx + 1) if a.ndim == 3 else (ny + 1, nx + 1)
    out = np.zeros(shape, dtype)
    out[tuple(slice(0, s) for s in a.shape)] = a
    return out


def fort(a):
    return np.ascontiguousarray(a.transpose(1, 0, 2) if a.ndim == 3 else a)


def run(exe, options, calls, frames, ny, nx, dtype, work):
    nz = frames[0]["t"].shape[0]
    inp, out = work / "input.bin", work / "output.bin"
    with inp.open("wb") as fh:
        def w(a, typ=dtype):
            fh.write(np.asarray(a, typ).tobytes())
        w([nx, ny, nz, len(calls)], np.int32)
        ints = [options["interval_m"], options["end_h"]]
        for prefix in ("if_no_pbl_nudging_", "if_zfac_", "k_zfac_", "dk_zfac_"):
            ints += [options[prefix + s] for s in SUFFIXES]
        ints += [options[k] for k in ("ktrop", "xwavenum", "ywavenum", "if_ramping", "stepfg", "fgdtzero")]
        w(ints, np.int32)
        w([options[k] for k in ("guv", "gt", "gph", "gq", "dtramp_min")] + [18.])
        for n in ("c1h", "c2h", "c1f", "c2f"):
            w(frames[0][n])
        for step, secs in calls:
            w([step, secs], np.int32)
            for frame, model in ((frames[0], True), (frames[1], False), (frames[2], False)):
                for name in FIELDS:
                    a = frame["th_phy"] if model and name == "t" else frame[name]
                    w(fort(rectangular(a, nz, ny, nx, dtype)))
            for name in ("pblh", "ht", "z_at_w"):
                w(fort(rectangular(frames[0][name], nz, ny, nx, dtype)))
            for name in ("mu_2", "mub", "muts"):
                w(fort(frames[0][name]))
    subprocess.run([str(exe), str(inp), str(out)], check=True, timeout=60)
    raw = np.fromfile(out, dtype).reshape(len(calls), 4, 5, ny + 1, nz + 1, nx + 1)
    shapes = ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz, ny, nx), (nz + 1, ny, nx), (nz, ny, nx))
    return {f"call{c}_{stage}{name}": raw[c, st, i].transpose(1, 0, 2)[:s[0], :s[1], :s[2]]
            for c in range(len(calls)) for st, stage in enumerate(("", "cpl_", "own_", "dec_"))
            for i, (name, s) in enumerate(zip(FIELDS, shapes))}


def main():
    build, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    opts = dict(guv=3e-4, gt=3e-4, gph=3e-4, gq=3e-4, interval_m=180, end_h=6,
                if_ramping=0, dtramp_min=0., ktrop=0, xwavenum=2, ywavenum=2, stepfg=1, fgdtzero=0)
    for s in SUFFIXES:
        opts.update({f"if_no_pbl_nudging_{s}": 0, f"if_zfac_{s}": 0,
                     f"k_zfac_{s}": 5, f"dk_zfac_{s}": 3})
    regimes = {
        "basic": {}, "dc": dict(xwavenum=0, ywavenum=0),
        "all_modes": dict(xwavenum=100, ywavenum=100),
        "negative_cut": dict(xwavenum=-1),
        "pbl_lid": {**{f"if_no_pbl_nudging_{s}": 1 for s in SUFFIXES},
                     **{f"if_zfac_{s}": 1 for s in SUFFIXES}, "ktrop": 30},
        "zfac": {f"if_zfac_{s}": 1 for s in SUFFIXES},
        "ramp_positive": dict(if_ramping=1, dtramp_min=60.),
        "ramp_negative": dict(if_ramping=1, dtramp_min=-60.),
        "hold": dict(stepfg=3), "zero_between": dict(stepfg=3, fgdtzero=1),
    }
    calls = [(1, 0), (2, 18), (301, 5400), (601, 10800), (1101, 19800),
             (1201, 21600), (1301, 23400), (1401, 25200), (1402, 25218)]
    manifest = {"reference": "original CPU-WRF Swiss history crops; full unmodified spectral_nudging + FFTPACK5",
                "sources_sha256": (build / "sources.sha256").read_text(), "cases": {}}
    for ny, nx in ((8, 10), (9, 11)):
        frames, paths = zip(*(original_frame(h, ny, nx) for h in (13, 12, 15)))
        after, _ = original_frame(14, ny, nx)
        frames[0]["muts"] = after["mu_2"] + after["mub"]
        for regime, edit in regimes.items():
            options = {**opts, **edit}
            seq = ([(k, (k - 1) * 18) for k in range(1, 7)]
                   if options["stepfg"] > 1 else calls)
            for kind, dtype in (("r4", np.float32), ("r8", np.float64)):
                name = f"swiss_{ny}x{nx}_{regime}_{kind}.npz"
                result = run(build / kind / "spectral_oracle", options, seq, frames, ny, nx, dtype, build)
                for i, frame in enumerate(frames):
                    result.update({f"frame{i}_{k}": np.asarray(v, dtype) for k, v in frame.items()})
                result["calls"] = np.asarray(seq, np.int32)
                np.savez_compressed(output / name, **result)
                manifest["cases"][name] = dict(
                    sha256=hashlib.sha256((output / name).read_bytes()).hexdigest(), options=options,
                    cpu_sources=[str(p) for p in paths],
                    nonzero={f: int(np.count_nonzero(result[f"call0_{f}"])) for f in FIELDS},
                    max_abs={f: float(np.abs(result[f"call0_{f}"]).max()) for f in FIELDS})
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
