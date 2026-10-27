"""Score changed W2 values against original WRF with frozen field/closure gates.

Field bounds come from retained energy/integration/water savepoint gates.
Water uses the pre-registered C27/C28 same-column REAL residual rule. The
placeholder noah_mp_step errwat=0 is NEVER used as conservation evidence.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Frozen: proofs/noahmp/{integration_step,energy_savepoint,water_savepoint}_gate.py
TOL = {"hfx": (2., .02), "lh": (3., .05), "grdflx": (2., .02), "tsk": (.3, 0.),
       "znt": (1.e-4, 0.), "emiss": (.002, 0.), "albedo": (.01, 0.),
       "smois": (5.e-6, 5.e-6), "sh2o": (5.e-6, 5.e-6)}
ENERGY_LIMIT = .05  # energy_savepoint_gate.TOL['erreng']


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cohort", choices=("warm", "cold", "frozen"), required=True)
    p.add_argument("--jarvis", action="store_true")
    p.add_argument("--floor", type=Path, help="manager-approved pristine 1-ulp ensemble floor.json")
    a = p.parse_args()
    sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
    import jax
    from columns import build, frozen_soil_columns
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
    from gpuwrf.surface_params import controls as sc
    assert jax.devices()[0].platform == "cpu"
    oracle = json.loads(a.oracle.read_text())
    floor = None if a.floor is None else json.loads(a.floor.read_text())
    if floor is not None:
        assert floor["oracle_sha256"] == hashlib.sha256(a.oracle.read_bytes()).hexdigest()
    cold = a.cohort == "cold"
    options = NoahMPOptions(opt_crs=2) if a.jarvis else None
    rows = frozen_soil_columns() if a.cohort == "frozen" else None
    land, forcing, static, ep, rp, fixture_nl, columns = build(cold=cold, options=options, rows=rows)
    if a.cohort == "frozen":
        assert [c["name"] for c in oracle["columns"]] == [c["name"] for c in columns]
    dt = float(columns[0]["dt"])
    names = ["rsmin"] if a.jarvis else sorted(set(sc.LSM_CONTROLS) - {"rsmin"})
    scored = [name for name in names if name in oracle["arms"]]  # frozen oracle: soil arms only
    base = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=dt,
        use_noahmp=True, noahmp_static=static, noahmp_land=land,
        noahmp_energy_params=ep, noahmp_rad_params=rp, noahmp_options=options)
    traces = []

    def program(l, f, nl, cb):
        traces.append(1)
        s, ep, rp = nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params
        values = sc.traced_controls(nl, cb)
        if values is not None:
            s, ep, rp = sc.apply_surface_controls(s, ep, rp, values)
        return noah_mp_step(l, f, s, dt, energy_params=ep, rad_params=rp,
                            options=nl.noahmp_options, return_diag=True)

    fn = jax.jit(program)
    prior = sc.prior_controls(fixture_nl, names)
    before = fn(land, forcing, base, build_clock_base(base))
    default_nl = dataclasses.replace(base, surface_controls=prior)
    defaults = fn(land, forcing, default_nl, build_clock_base(default_nl))
    before_host = [np.asarray(x) for x in jax.tree.leaves(before)]
    default_host = [np.asarray(x) for x in jax.tree.leaves(defaults)]
    default_bytes = sum(x.tobytes() == y.tobytes() for x, y in zip(before_host, default_host, strict=True))
    results = {"source": str(a.repo.resolve()), "platform": "cpu", "gpu_executed": False,
               "oracle_sha256": hashlib.sha256(a.oracle.read_bytes()).hexdigest(), "cohort": a.cohort,
               "jarvis": a.jarvis, "names": names, "scored": scored, "columns": [c["name"] for c in columns],
               "tolerances": TOL, "energy_limit_Wm2": ENERGY_LIMIT,
               "water_rule": "abs(native residual)<=1.5*abs(pristine REAL residual) AND <.1mm, same initial inputs",
               "default_equal_leaves": default_bytes, "output_leaves": len(default_host), "arms": {}}
    if floor is not None:
        results.update(water_rule=floor["binding_rule"], water_floor_sha256=hashlib.sha256(a.floor.read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    start = 4 if cold else 0
    keys = []
    for name in scored:
        ref = oracle["arms"][name]
        factor = float(ref["factor"])
        controls = sc.expand_controls(fixture_nl, {key: factor if key == name else 1. for key in names})
        nl = dataclasses.replace(base, surface_controls=controls)
        cb = build_clock_base(nl)
        key = ck.cheap_key(fn, (land, forcing, nl, cb), {}, nl)
        assert key is not None
        keys.append(key)
        new_land, flux, diagnostic = fn(land, forcing, nl, cb)
        actual = jax.device_get((new_land, flux, diagnostic))
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(actual)), (name, "non-finite candidate")
        current = [np.asarray(x) for x in jax.tree.leaves(actual)]
        row = {"factor": factor, "cheap_key": key, "cache_size": fn._cache_size(), "fields": [], "closure": [],
               "changed_leaves": sum(x.tobytes() != y.tobytes() for x, y in zip(default_host, current, strict=True))}
        for j, column in enumerate(columns):
            wrf = ref["oracle"][start + j]
            driver, energy = wrf["driver"], wrf["energy_out"]
            # Pristine noahmpdrv:1230-1232 exports SALB only when > -999;
            # nighttime SFLX -999.9 means retain the caller's previous ALBEDO.
            raw_albedo = wrf["energy_state"]["albedo"]
            albedo = raw_albedo if raw_albedo > -999 else float(np.asarray(land.albedo)[0, j])
            targets = {"hfx": driver["hfx"], "lh": driver["lh"], "grdflx": driver["grdflx"], "tsk": driver["tsk"],
                       "znt": energy["z0wrf"], "emiss": energy["emissi"],
                       "albedo": albedo, "smois": wrf["smc_out"], "sh2o": wrf["sh2o_out"]}
            for field, expected in targets.items():
                source = actual[0] if field in ("smois", "sh2o") else actual[1]
                got = np.asarray(getattr(source, field))[..., j].reshape(-1).astype(np.float64)
                expected = np.asarray(expected, np.float64).reshape(-1)
                assert got.shape == expected.shape and np.isfinite(expected).all()
                atol, rtol = TOL[field]
                error = np.abs(got - expected)
                row["fields"].append({"column": column["name"], "field": field, "max_abs": float(error.max()),
                                      "candidate": got.tolist(), "reference": expected.tolist(),
                                      "pass": bool(np.all(error <= atol + rtol * np.abs(expected)))})
            dz = np.asarray(static.dzs, np.float64).reshape(4)
            initial = column["state_in"]
            initial_mass = float(np.dot(np.asarray(initial["smc"]), dz) * 1000 + initial["sneqv"]
                                 + initial["canliq"] + initial["canice"])
            new_mass = float(np.dot(np.asarray(actual[0].smois)[:, 0, j].astype(np.float64), dz) * 1000
                             + actual[0].sneqv[0, j] + actual[0].canliq[0, j] + actual[0].canice[0, j])
            wrf_mass = float(np.dot(np.asarray(wrf["smc_out"]), dz) * 1000 + wrf["snow_out"]["sneqv"]
                             + wrf["water_out"]["canliq"] + wrf["water_out"]["canice"])
            # Precipitation input (mm); zero on every dry column, so dry receipts are unchanged.
            precipitation = (column["forcing"]["prcpconv"] + column["forcing"]["prcpnonc"]) * dt
            native_residual = abs(new_mass - initial_mass + float(actual[1].qfx[0, j]) * dt
                                  + 1000 * (float(actual[0].sfcrunoff[0, j]) + float(actual[0].udrunoff[0, j]))
                                  - precipitation)
            wrf_residual = abs(wrf_mass - initial_mass + driver["qfx"] * dt
                               + wrf["water_out"]["runsrf"] + wrf["water_out"]["runsub"] - precipitation)
            energy_native = abs(float(actual[2].erreng[0, j]))
            energy_wrf = abs(ref["closure"][start + j][0])
            absolute_floor = 0. if floor is None else next(r["floor_mm"] for r in floor["arms"][name]["columns"]
                                                         if r["name"] == column["name"])
            assert np.isfinite(absolute_floor) and absolute_floor >= 0.
            water_limit = max(1.5 * wrf_residual, absolute_floor)
            row["closure"].append({"column": column["name"], "native_water_mm": native_residual,
                "pristine_water_mm": wrf_residual, "water_floor_mm": absolute_floor, "water_limit_mm": min(water_limit, .1),
                "water_pass": bool(native_residual <= water_limit and native_residual < .1),
                "native_energy_Wm2": energy_native, "pristine_energy_Wm2": energy_wrf,
                "energy_pass": bool(energy_native <= ENERGY_LIMIT and energy_wrf <= ENERGY_LIMIT)})
        row["field_pass"] = all(x["pass"] for x in row["fields"])
        row["closure_pass"] = all(x["water_pass"] and x["energy_pass"] for x in row["closure"])
        results["arms"][name] = row
        a.out.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"name": name, "field_pass": row["field_pass"], "closure_pass": row["closure_pass"],
                          "changed_leaves": row["changed_leaves"], "cache_size": row["cache_size"]}), flush=True)
    results["one_key"] = len(set(keys)) == 1
    results["traces"] = len(traces)
    results["pass"] = results["one_key"] and len(traces) == 2 and all(r["field_pass"] and r["closure_pass"]
                                                                 for r in results["arms"].values())
    a.out.write_text(json.dumps(results, indent=2) + "\n")
    return int(not results["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
