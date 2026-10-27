"""Measure the actual pristine WRF SINTB -> boundary -> radiation input chain.

This is separate from the isolated injected-parent operator oracle: bdy_interp1
stores old CHILD values in BXS and parent differences in BTXS. Radiation only
reads BXS. No port implementation participates in this experiment.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from oracle import build, extract, load, oracle, sha


def build_boundary(wrf_root, out, fc):
    source = (Path(__file__).parent / "caller_wrapper.F90").read_text().replace(
        "! PRISTINE_BDY", extract(wrf_root / "share/interp_fcn.F", "bdy_interp1")).replace(
        "! PRISTINE_SPEC", extract(wrf_root / "share/module_bc.F", "spec_bdyfield"))
    source += "\n" + (wrf_root / "share/sint.F").read_text()
    path = out / "caller.F90"
    path.write_text(source)
    so = out / "caller.so"
    subprocess.run([fc, "-shared", "-fPIC", "-O2", "-ffp-contract=off", "-fcheck=bounds",
                    "-ffree-form", "-ffree-line-length-none", str(path), "-o", str(so)],
                   cwd=out, check=True, timeout=120)
    fn = ctypes.CDLL(str(so)).caller_probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int] * 7 + [arr] * 4
    fn.restype = None
    return fn, path, so


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wrf-root", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    p.add_argument("--case", type=Path, default=Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227"))
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lib, receipt = build(args.wrf_root, args.out)
    fn, path, so = build_boundary(args.wrf_root, args.out, receipt["compiler"])
    parent, child = [load(args.case / f"wrfinput_{d}") for d in ("d02", "d03")]
    py, px = parent["HGT"].shape
    ny, nx = child["HGT"].shape
    receipt["cases"] = []
    for hour in (7.5, 18.0):
        _, ph, _ = oracle(lib, parent, hour * 60)
        _, ch, _ = oracle(lib, child, hour * 60)
        c = np.asfortranarray(np.pad(ch, ((0, 1), (0, 1)), mode="edge").T[:, None, :])
        results = []
        for offset in (0., 1000.):
            # Uniform 1000m parent-height perturbation is an explicit causal
            # control on the actual real-structured parent shadow field.
            pp = np.asfortranarray(np.pad(ph + np.float32(offset), 3, mode="edge").T[:, None, :])
            out = np.zeros_like(c, order="F")
            delta = np.zeros(4, np.float32)
            fn(px, py, nx, ny, child["ipos"], child["jpos"], child["ratio"], pp, c, out, delta)
            assert np.isfinite(out).all() and np.isfinite(delta).all()
            assert np.array_equal(out[:nx, :, :ny], c[:nx, :, :ny]), "BXS did not retain child's height"
            assert delta.max() > 0, "parent did not reach the boundary builder"
            results.append((out.copy(), delta.copy()))
        assert np.array_equal(results[0][0], results[1][0]), "parent perturbation reached radiation BXS"
        assert not np.array_equal(results[0][1], results[1][1]), "parent perturbation did not change BTXS"
        row = {"hour_utc": hour, "spec_bdyfield_vs_old_child_max_m": float(np.max(np.abs(results[0][0] - c))),
               "parent_plus1000m_effect_on_radiation_input_m": float(np.max(np.abs(results[0][0] - results[1][0]))),
               "boundary_tendency_max_m_per_s": results[0][1].tolist(),
               "parent_plus1000m_tendency_max_m_per_s": results[1][1].tolist()}
        receipt["cases"].append(row)
        print(json.dumps(row), flush=True)
    receipt.update(status="PASS", conclusion="SINTB parent changes only BTXS; radiation BXS equals old child HGT_SHAD",
                   source_sha256=sha(path), caller_library_sha256=sha(so),
                   interp_source_sha256=sha(args.wrf_root / "share/interp_fcn.F"),
                   spec_source_sha256=sha(args.wrf_root / "share/module_bc.F"),
                   sint_source_sha256=sha(args.wrf_root / "share/sint.F"))
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
