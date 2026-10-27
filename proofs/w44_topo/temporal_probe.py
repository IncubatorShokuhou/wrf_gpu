"""Actual init -> SINTB -> spec_bdyfield -> toposhad chain over a full day.

Uses unmodified pristine routines, real WN3 grids and the source driver order:
nests open before their parent's first solve, so cold-start HGT_SHAD is zero.
Parent field perturbations after opening are an explicit causal control.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
import jax
import jax.numpy as jnp
import numpy as np

from oracle import build, load, oracle, sha
from caller_probe import build_boundary
from gpuwrf.coupling.physics_couplers import _wrf_topographic_shadow_mask, _compute_solar_geometry
from operator_probe import topographic_shadow


def force(fn, pf, cf, parent_height, child_height):
    py, px = pf["HGT"].shape
    ny, nx = cf["HGT"].shape
    F = lambda a: np.asfortranarray(a.T[:, None, :])
    pp = F(np.pad(parent_height, 3, mode="edge"))
    cc = F(np.pad(child_height, ((0, 1), (0, 1)), mode="edge"))
    out, tendency = np.zeros_like(cc, order="F"), np.zeros(4, np.float32)
    fn(px, py, nx, ny, cf["ipos"], cf["jpos"], cf["ratio"], pp, cc, out, tendency)
    return out[:nx, 0, :ny].T.copy(), tendency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--native-interpret", action="store_true")
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    case = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227")
    lib, receipt = build(wrf, args.out)
    fn, source, so = build_boundary(wrf, args.out, receipt["compiler"])
    spec = importlib.util.spec_from_file_location("pristine_init", root / "proofs/s1_nestspec/pristine_nest_start.py")
    init = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(init)
    init_lib = init.build(args.out / "init")
    fields = {d: load(case / f"wrfinput_{d}") for d in ("d01", "d02", "d03")}
    heights = {"d01": np.zeros_like(fields["d01"]["HGT"])}
    for parent, child in (("d01", "d02"), ("d02", "d03")):
        cf = fields[child]
        heights[child] = init.interpolate(init_lib, heights[parent][None, ...], child_shape=cf["HGT"].shape,
                                           start=(cf["ipos"], cf["jpos"]), ratio=cf["ratio"])[0]
        assert np.array_equal(heights[child], np.zeros_like(heights[child]))
    receipt["init_order"] = "module_integrate.F:337-351 opens nests before solve_interface:393; ht_shad allocator initial value 0; init interp_fcn_sint verbatim"
    receipt["inputs"] = {d: {"path": str(case / f"wrfinput_{d}"), "sha256": sha(case / f"wrfinput_{d}")} for d in fields}
    receipt["calls"] = []
    candidates = {}
    for domain in ("d02", "d03"):
        f = fields[domain]
        def one(terrain, old_height, declin, minute, f=f):
            if args.native_interpret:
                from gpuwrf.physics.topographic_shading import toposhad_solar_geometry, toposhad_solar_azimuth
                from gpuwrf.kernels.topographic_shadow import topographic_shadow_native
                csza, hrang = toposhad_solar_geometry(f["XLAT"], f["XLONG"], declin, minute, 0., 30.)
                azi = toposhad_solar_azimuth(f["XLAT"], csza, declin, hrang, f["SINALPHA"], f["COSALPHA"])
                return topographic_shadow_native(terrain, old_height, csza, azi,
                    dx_m=f["dx"], dy_m=f["dy"], shadow_length_m=25000., nested=True, interpret=True)
            return topographic_shadow(terrain, parent_shadow_height_m=old_height,
                latitude_deg=f["XLAT"], longitude_deg=f["XLONG"], sina=f["SINALPHA"], cosa=f["COSALPHA"],
                declination_rad=declin, xtime_minutes=minute, gmt_hours=0., radt_minutes=30.,
                dx_m=f["dx"], dy_m=f["dy"], shadow_length_m=25000.)
        candidates[domain] = jax.jit(one)
    retained = {d: h.copy() for d, h in heights.items()}
    ports = {}
    for domain in ("d02", "d03"):
        f = fields[domain]
        ports[domain] = jax.jit(lambda csza, declin, hrang, f=f: _wrf_topographic_shadow_mask(
            f["HGT"], latitude_deg=f["XLAT"], coszen=csza, declination_rad=declin,
            hour_angle_rad=hrang, sina=f["SINALPHA"], cosa=f["COSALPHA"],
            dx_m=f["dx"], dy_m=f["dy"], shadow_length_m=25000.))
    for minute in range(0, args.hours * 60 + 1, 30):
        F = np.float32
        day = F(58) + F(minute) / F(1440)
        sxday = day - F(80) if day >= F(80) else day + F(285)
        declin = F(np.arcsin(np.sin(F(23.5) * F(np.pi / 180)) * np.sin(F(360) / F(365) * sxday * F(np.pi / 180))))
        _, heights["d01"], _ = oracle(lib, fields["d01"], minute, declin_override=declin)
        for parent, child in (("d01", "d02"), ("d02", "d03")):
            f, previous = fields[child], heights[child]
            supplied, tendency = force(fn, fields[parent], f, heights[parent], previous)
            perturbed, _ = force(fn, fields[parent], f, heights[parent] + np.float32(1000), previous)
            assert np.array_equal(supplied, previous) and np.array_equal(supplied, perturbed)
            mask, height, declin = oracle(lib, f, minute, supplied, declin_override=declin)
            retained_mask, retained_height = candidates[child](f["HGT"], retained[child], declin, np.float32(minute))
            assert np.isfinite(height).all() and np.isfinite(np.asarray(retained_height)).all()
            # This is a composed multi-call test: candidate receives its OWN
            # previous height, never a fresh Fortran value after initialization.
            # Complete the reference diagnosis even if the experimental
            # candidate fails. Preserve the registered strict verdict below.
            mask_errors = int(np.count_nonzero(np.asarray(retained_mask) != mask))
            height_errors = int(np.count_nonzero(np.abs(np.asarray(retained_height) - height) >
                                                .01 + 2.e-5 * np.abs(height)))
            retained[child] = np.asarray(retained_height)
            rad = np.float32(np.pi / 180)
            hrang = np.float32(15) * (np.float32(minute / 60 + .25) + f["XLONG"] / np.float32(15) - np.float32(12)) * rad
            lat = f["XLAT"] * rad
            coszen = np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.cos(hrang)
            with jax.enable_x64(False):
                local_mask = np.asarray(ports[child](coszen, declin, hrang))
            # Report the port's actual EOT-bearing solar geometry separately;
            # the same-geometry arm isolates the missing memory/halo semantics.
            geom = _compute_solar_geometry(f["XLAT"], f["XLONG"], datetime(2026, 2, 28, tzinfo=timezone.utc),
                                           minute * 60, solar_lead_seconds=(minute + 15) * 60)
            with jax.enable_x64(False):
                actual_mask = np.asarray(ports[child](geom.coszen, geom.declination_rad, geom.hour_angle_rad))
            row = {"domain": child, "minute": minute, "shadow_cells": int(mask.sum()),
                   "BXS_vs_previous_child_max_m": float(np.max(np.abs(supplied - previous))),
                   "parent_plus1000m_effect_m": float(np.max(np.abs(supplied - perturbed))),
                   "candidate_height_max_error_m": float(np.max(np.abs(np.asarray(retained_height) - height))),
                   "candidate_mask_misses": mask_errors,
                   "candidate_height_outside_registered_tolerance": height_errors,
                   "released_port_same_geometry_mask_misses": int(np.count_nonzero(local_mask != mask)),
                   "released_port_actual_geometry_mask_misses": int(np.count_nonzero(actual_mask != mask))}
            receipt["calls"].append(row)
            heights[child] = height
            print(json.dumps(row), flush=True)
    passed = all(r["candidate_mask_misses"] == 0 and r["candidate_height_outside_registered_tolerance"] == 0
                 for r in receipt["calls"])
    receipt.update(status="PASS" if passed else "FAIL", reference_chain_complete=True,
                   calls_per_nest=args.hours * 2 + 1, caller_source_sha256=sha(source), caller_so_sha256=sha(so),
                   candidate="pallas CPU interpretation" if args.native_interpret else "JAX CPU reference")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
