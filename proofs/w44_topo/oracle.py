"""Build verbatim pristine-WRF topographic shading routines and compare REAL operands.

CPU only. All inputs are from the project's admitted staged WN3 0227 copy.
No ALISIOS paths or WRF forecast outputs are read. Build products stay outside git.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract(path, routine):
    source = path.read_text()
    match = re.search(r"^\s*SUBROUTINE " + routine + r"\s*\(.*?^\s*END SUBROUTINE " + routine + r"\s*$",
                      source, re.I | re.M | re.S)
    if match is None:
        raise ValueError(f"missing pristine routine {routine}")
    return match.group()


def build(wrf, out):
    fc = os.environ.get("FC") or shutil.which("gfortran")
    if not fc:
        fallback = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
        if fallback.exists():
            fc = str(fallback)
    if not fc:
        raise RuntimeError("set FC to the WRF Fortran compiler")
    radiation = wrf / "phys/module_radiation_driver.F"
    surface = wrf / "phys/module_surface_driver.F"
    routines = extract(radiation, "toposhad_init") + "\n" + extract(radiation, "toposhad")
    solar_lines = "     TLOCTM=GMT" + extract(radiation, "toposhad").split(
        "     TLOCTM=GMT", 1)[1].split("     if (csza.lt.", 1)[0]
    wrapper = (HERE / "oracle_wrapper.F90").read_text()
    source = out / "oracle.F90"
    source.write_text(wrapper.replace("! PRISTINE_SHADING", routines).replace(
        "! PRISTINE_FLUX", extract(surface, "TOPO_RAD_ADJ")).replace(
        "! PRISTINE_CLOCK", extract(radiation, "radconst") + "\n" + extract(radiation, "calc_coszen")).replace(
        "! PRISTINE_TOPOSOLAR", solar_lines))
    library = out / "oracle.so"
    command = [fc, "-shared", "-fPIC", "-O2", "-ffp-contract=off", "-ffree-form",
               "-ffree-line-length-none", str(source), "-o", str(library)]
    subprocess.run(command, cwd=out, check=True, timeout=120)
    lib = ctypes.CDLL(str(library))
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
    lib.shadow_probe.argtypes = [ctypes.c_int] * 3 + [arr] * 6 + [ctypes.c_float] * 7 + [ints, arr]
    lib.flux_probe.argtypes = [ctypes.c_int] + [arr] * 8 + [ints, arr]
    return lib, {"compiler": fc, "command": command, "library_sha256": sha(library),
                 "radiation_source_sha256": sha(radiation), "surface_source_sha256": sha(surface),
                 "extracted_source_sha256": sha(source), "wrapper_sha256": sha(HERE / "oracle_wrapper.F90")}


def load(path):
    with netCDF4.Dataset(path) as ds:
        fields = {n: np.asarray(ds[n][0], np.float32) for n in
                  ("HGT", "XLAT", "XLONG", "SINALPHA", "COSALPHA")}
        fields.update(dx=float(ds.DX), dy=float(ds.DY),
                      ipos=int(ds.I_PARENT_START), jpos=int(ds.J_PARENT_START), ratio=int(ds.PARENT_GRID_RATIO))
    return fields


def oracle(lib, fields, xtime, parent=None, *, declin_override=None):
    F = lambda a: np.asfortranarray(np.asarray(a, np.float32).T)
    ny, nx = fields["HGT"].shape
    halo = np.pad(fields["HGT"], 2, mode="edge")
    full = lambda a: F(np.pad(a, 2, mode="edge"))
    height = F(np.pad(fields["HGT"] - np.float32(.001) if parent is None else parent, 2, mode="edge"))
    mask = np.zeros((nx + 4, ny + 4), np.int32, order="F")
    # Feb 28 morning/evening; radconst's REAL declination, no equation-of-time in toposhad.
    declin = np.float32(np.arcsin(np.sin(np.float32(23.5) * np.float32(np.pi / 180)) *
                                np.sin(np.float32(360 / 365) * np.float32(58 - 80) * np.float32(np.pi / 180))))
    if declin_override is not None:
        declin = np.float32(declin_override)
    initial_height = height.copy(order="F")
    lib.shadow_probe(nx, ny, int(parent is not None), F(halo), full(fields["XLAT"]),
                     full(fields["XLONG"]), full(fields["SINALPHA"]), full(fields["COSALPHA"]), initial_height,
                     xtime, 0., 30., float(declin), fields["dx"], fields["dy"], 25000., mask, height)
    return mask.T[2:-2, 2:-2].copy(), height.T[2:-2, 2:-2].copy(), declin


def compare(name, got, ref, *, atol, rtol=0):
    got, ref = np.asarray(got), np.asarray(ref)
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise AssertionError(f"{name}: nonfinite candidate or reference")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=name)
    return float(np.max(np.abs(got - ref)))


def main():
    import jax
    import jax.numpy as jnp
    from operator_probe import topographic_shadow
    from gpuwrf.nesting.interp import build_sint_weights, interp_sint_full
    from gpuwrf.coupling.physics_couplers import wrf_radiation_slope_aspect_from_terrain
    from gpuwrf.physics.rrtmg_sw import wrf_topographic_sw_correction_factor
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wrf-root", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    p.add_argument("--case", type=Path, default=Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227"))
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == "cpu"
    lib, receipt = build(args.wrf_root, args.out)
    fields = {d: load(args.case / f"wrfinput_{d}") for d in ("d02", "d03")}
    receipt["inputs"] = {d: {"path": str(args.case / f"wrfinput_{d}"), "sha256": sha(args.case / f"wrfinput_{d}")}
                         for d in fields}
    receipt["tolerances_registered"] = {"mask": "exact", "height_m": {"atol": .01, "rtol": 2.e-5},
                                          "flux_wm2": {"atol": .002, "rtol": 2.e-5}}
    receipt["cases"] = []
    fixtures = {}
    child = fields["d03"]
    ny, nx = child["HGT"].shape
    parent_field = fields["d02"]
    weights = build_sint_weights(parent_grid_ratio=child["ratio"], i_parent_start=child["ipos"],
                                j_parent_start=child["jpos"], parent_ny=parent_field["HGT"].shape[0],
                                parent_nx=parent_field["HGT"].shape[1], child_ny=ny, child_nx=nx)
    # Morning/evening and night: angles fixed before comparison, no tuned cell selection.
    for hour in (7.5, 8.0, 9.0, 17.0, 18.0, 23.0):
        xtime = hour * 60
        _, ph, declin = oracle(lib, parent_field, xtime)
        parent = np.asarray(interp_sint_full(jnp.asarray(ph), weights, parent_grid_ratio=child["ratio"]))
        ref_mask, ref_height, _ = oracle(lib, child, xtime, parent)
        old_mask, _, _ = oracle(lib, child, xtime)
        fn = jax.jit(lambda terrain, parent: topographic_shadow(
            terrain, latitude_deg=child["XLAT"], longitude_deg=child["XLONG"], sina=child["SINALPHA"],
            cosa=child["COSALPHA"], declination_rad=declin, xtime_minutes=xtime, gmt_hours=0.,
            radt_minutes=30., dx_m=child["dx"], dy_m=child["dy"], shadow_length_m=25000.,
            parent_shadow_height_m=parent))
        got_mask, got_height = fn(jnp.asarray(child["HGT"]), jnp.asarray(parent))
        row = {"hour_utc": hour, "shape": [ny, nx], "shadow_cells": int(ref_mask.sum()),
               "no_parent_mutant_mask_misses": int(np.count_nonzero(old_mask != ref_mask))}
        row["mask_max_error"] = compare("mask", got_mask, ref_mask, atol=0)
        row["height_max_error_m"] = compare("height", got_height, ref_height, atol=.01, rtol=2.e-5)
        # Same WRF surface caller operands on real slopes. This isolates the
        # mask's flux consequence; horizontal SWDOWN stays 600 in both arms.
        slope, aspect = wrf_radiation_slope_aspect_from_terrain(
            child["HGT"], dx_m=child["dx"], dy_m=child["dy"])
        slope, aspect = np.asarray(slope, np.float32), np.asarray(aspect, np.float32)
        rad = np.float32(np.pi / 180)
        hrang = np.float32(15) * (np.float32(hour + .25) + child["XLONG"] / np.float32(15) - np.float32(12)) * rad
        csza = np.sin(child["XLAT"] * rad) * np.sin(declin) + np.cos(child["XLAT"] * rad) * np.cos(declin) * np.cos(hrang)
        arrays = [child["XLAT"], child["XLONG"], csza, np.full_like(csza, .3),
                  np.full_like(csza, declin), hrang, slope, aspect]
        flat = lambda a: np.asfortranarray(a.reshape(-1))
        flux = np.zeros(nx * ny, np.float32)
        lib.flux_probe(nx * ny, *[flat(a) for a in arrays], flat(ref_mask), flux)
        factor = jax.jit(wrf_topographic_sw_correction_factor)(
            jnp.asarray(csza), jnp.asarray(arrays[3]), jnp.asarray(child["XLAT"]),
            jnp.asarray(declin), jnp.asarray(hrang), jnp.asarray(slope), jnp.asarray(aspect), got_mask)
        row["swnorm_max_error_wm2"] = compare("SWNORM", np.asarray(factor) * np.float32(600),
                                              flux.reshape(ny, nx), atol=.002, rtol=2.e-5)
        receipt["cases"].append(row)
        for key, a in (("terrain", child["HGT"]), ("latitude", child["XLAT"]), ("longitude", child["XLONG"]),
                       ("sina", child["SINALPHA"]), ("cosa", child["COSALPHA"]), ("parent_height", parent),
                       ("mask", ref_mask), ("height", ref_height)):
            fixtures[f"h{hour}_{key}"] = a
        fn.clear_cache()
        print(json.dumps(row), flush=True)
    assert sum(r["no_parent_mutant_mask_misses"] for r in receipt["cases"]) > 0, "vacuous parent test"
    np.savez_compressed(args.out / "fixture.npz", **fixtures)
    receipt["fixture_sha256"] = sha(args.out / "fixture.npz")
    receipt["status"] = "PASS"
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
