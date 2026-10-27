"""Pristine solar clocks: RRTMG/slope use EOT; toposhad does not.

Flux effects use real terrain/slopes with explicit 600 W/m2 horizontal SWDOWN
and diffuse fraction .3. They are operator measurements, not forecast fluxes.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
import jax
import jax.numpy as jnp
import numpy as np

from oracle import build, compare, load, oracle, sha
from gpuwrf.coupling import physics_couplers as port
from gpuwrf.physics.rrtmg_sw import wrf_topographic_sw_correction_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    case = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227")
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    lib, receipt = build(wrf, args.out)
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    lib.solar_probe.argtypes = [ctypes.c_int] * 2 + [arr] * 2 + [ctypes.c_float] * 4 + [arr] * 5
    lib.solar_probe.restype = None
    receipt.update(cases=[], bounds={"coszen": 2.e-6, "hour_angle_rad": 3.e-6, "declination_rad": 2.e-7,
                                     "flux_wm2": {"atol": .002, "rtol": 2.e-5}},
                   solar_julian_wrf_flag=bool(port._SOLAR_JULIAN_WRF),
                   horizontal_swdown_wm2=600, diffuse_fraction=.3,
                   scope="isolated surface operator; no full RRTMG flux/forecast measurement")
    assert port._SOLAR_JULIAN_WRF, "not the released solar clock"
    for domain in ("d02", "d03"):
        f = load(case / f"wrfinput_{domain}")
        ny, nx = f["HGT"].shape
        F = lambda a: np.asfortranarray(np.asarray(a, np.float32).T)
        slope, aspect = port.wrf_radiation_slope_aspect_from_terrain(f["HGT"], dx_m=f["dx"], dy_m=f["dy"])
        slope, aspect = np.asarray(slope, np.float32), np.asarray(aspect, np.float32)
        with jax.enable_x64(False):
            solar = jax.jit(lambda lead: port._compute_solar_geometry(
                f["XLAT"], f["XLONG"], datetime(2026, 2, 28, tzinfo=timezone.utc), lead,
                solar_lead_seconds=lead + jnp.float32(900)))
            mask_fn = jax.jit(lambda coszen, declin, hrang: port._wrf_topographic_shadow_mask(
                f["HGT"], latitude_deg=f["XLAT"], coszen=coszen, declination_rad=declin,
                hour_angle_rad=hrang, sina=f["SINALPHA"], cosa=f["COSALPHA"],
                dx_m=f["dx"], dy_m=f["dy"], shadow_length_m=25000.))
        for day in (0, 1, 2):
            for hour in (0., 7.5, 8., 12., 17., 18., 18.5):
                minute = day * 1440 + hour * 60
                declin = np.zeros(1, np.float32)
                outputs = [np.zeros((nx, ny), np.float32, order="F") for _ in range(4)]
                lib.solar_probe(nx, ny, F(f["XLAT"]), F(f["XLONG"]), np.float32(58 + minute / 1440),
                                minute, 0., 30., declin, *outputs)
                csza, hrang, shade_csza, shade_hrang = [x.T.copy() for x in outputs]
                with jax.enable_x64(False):
                    geom = solar(jnp.float32(minute * 60))
                    old_mask = np.asarray(mask_fn(geom.coszen, geom.declination_rad, geom.hour_angle_rad))
                    fixed_clock_mask = np.asarray(mask_fn(shade_csza, declin[0], shade_hrang))
                ref_mask, _, _ = oracle(lib, f, minute, declin_override=declin[0])
                row = dict(domain=domain, day=day, hour=hour)
                row["rrtmg_coszen_max_error"] = compare("RRTMG coszen", geom.coszen, csza, atol=2.e-6)
                row["slope_hrang_max_error_rad"] = compare("slope hrang", geom.hour_angle_rad, hrang, atol=3.e-6)
                row["declin_max_error_rad"] = compare("declin", geom.declination_rad, declin[0], atol=2.e-7)
                row["toposhad_clock_max_csza_difference"] = float(np.max(np.abs(csza - shade_csza)))
                row["old_mask_misses"] = int(np.count_nonzero(old_mask != ref_mask))
                row["separate_toposhad_clock_mask_misses"] = int(np.count_nonzero(fixed_clock_mask != ref_mask))
                flat = lambda a: np.asfortranarray(np.asarray(a).reshape(-1))
                arrays = [f["XLAT"], f["XLONG"], csza, np.full_like(csza, .3),
                          np.full_like(csza, declin[0]), hrang, slope, aspect]
                reference_flux = np.zeros(nx * ny, np.float32)
                lib.flux_probe(nx * ny, *[flat(a) for a in arrays], flat(ref_mask), reference_flux)
                def surface(mask):
                    return np.asarray(wrf_topographic_sw_correction_factor(
                        *[jnp.asarray(a) for a in (csza, arrays[3], f["XLAT"], declin[0], hrang, slope, aspect, mask)]),
                        np.float32) * np.float32(600)
                corrected_flux = surface(fixed_clock_mask)
                row["correct_mask_swnorm_max_error_wm2"] = compare("correct SWNORM", corrected_flux,
                    reference_flux.reshape(ny, nx), atol=.002, rtol=2.e-5)
                error = surface(old_mask) - reference_flux.reshape(ny, nx)
                row["old_swnorm_max_error_wm2"] = float(np.max(np.abs(error)))
                row["old_swnorm_abs_error_sum_wm2"] = float(np.abs(error).sum())
                receipt["cases"].append(row)
                print(json.dumps(row), flush=True)
        solar.clear_cache()
        mask_fn.clear_cache()
    receipt["status"] = "PASS"
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
