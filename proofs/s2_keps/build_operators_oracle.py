"""Build small REAL32 subroutine fixtures from unchanged private WRF sources.

The fixture boundary is each named helper routine, with all operands explicit;
it does not qualify EEPS2D/KEPS or their operational caller. All numeric gates
are fixed in the test before comparison: 16 float32 ulps at each output field's
scale, exact zeros, finite candidate AND reference. Transcendentals and chained
single-precision arithmetic justify the ulp budget. No GPU speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


FIELDS = ("u", "v", "theta", "theta0", "dz", "rho", "tke", "epsilon", "tpe", "prandtl",
          "gradient", "nsquare", "richardson", "epsilon_source", "lower", "diagonal", "upper",
          "rhs", "volume", "surface", "prandtl_face", "rho_face", "diffusivity", "length",
          "analytic_shear", "analytic_buoy")


def cases():
    """Adversarial REAL profiles: stable/unstable/neutral, rescue, nonuniform dz."""
    records = []
    for n in (8, 12, 44):
        for regime in ("stable", "unstable", "neutral", "rescue", "nonuniform", "exchange_floor"):
            k = np.arange(n, dtype=np.float32)
            dz = (35.+k*7. if regime == "nonuniform" else np.full(n, 100., np.float32)).astype(np.float32)
            g = {"u": 3.+.1*k+.001*k**2, "v": -1.+.02*k-.0003*k**2,
                 "theta": 290.+(.15 if regime == "stable" else -.06)*k,
                 "theta0": np.full(n, 300.), "dz": dz, "rho": 1.2-.008*k,
                 "tke": .05+.002*k, "epsilon": .003+.00002*k,
                 "tpe": .004+.0001*k, "prandtl": 1.+.03*k,
                 "gradient": .0001*(k-n/2), "nsquare": -.001*(k%3),
                 "richardson": .1*(k%3-1), "epsilon_source": -.0001-.00001*k,
                 "lower": -.05-.001*k, "diagonal": 1.2+.003*k, "upper": -.08+.0002*k,
                 "rhs": .5+.1*k, "volume": .8+.003*k,
                 "length": 20.+5.*k, "analytic_shear": np.full(n, .00003),
                 "analytic_buoy": np.full(n, -.0001 if regime == "stable" else .0002)}
            if regime == "neutral":
                g["analytic_shear"] = np.zeros(n)
                g["analytic_buoy"] = np.zeros(n)
            if regime == "rescue":
                g["epsilon"] = np.full(n, 2.e-7)
                g["tke"] = np.full(n, .1)
            if regime == "exchange_floor":
                g["tke"] = np.full(n, 1.e-4)
                g["epsilon"] = np.full(n, .01)
            g = {name: np.asarray(value, np.float32) for name, value in g.items()}
            g["surface"] = np.linspace(.6, 1., n+1, dtype=np.float32)
            g["prandtl_face"] = np.r_[np.float32(0.), .5*(g["prandtl"][:-1]+g["prandtl"][1:]), g["prandtl"][-1]]
            g["rho_face"] = np.r_[g["rho"][0], (g["rho"][1:]*dz[:-1]+g["rho"][:-1]*dz[1:])/(dz[:-1]+dz[1:]), g["rho"][-1]]
            g["diffusivity"] = np.r_[np.float32(0.), np.linspace(.1, .5, n, dtype=np.float32)]
            records.append({"name": f"{regime}_{n}", "n": n, "dt": 1. if regime == "rescue" else 18.,
                            "roughness": .1, "ustar": .3, "phi_epsilon": 1.2, "phi_m": 1.1,
                            "input": {name: values.tolist() for name, values in g.items()}})
    return records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrf-root", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    args = ap.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    args.build.mkdir(parents=True, exist_ok=True)
    flags = ["-ffree-form", "-ffree-line-length-none", "-O0", "-ffp-contract=off", "-fcheck=all"]
    sources = {}
    objects = []
    for rel in ("share/module_model_constants.F", "phys/module_bl_keps.F", "phys/module_bl_eepsilon.F"):
        source = args.wrf_root / rel
        dest = args.build / source.name
        shutil.copyfile(source, dest)
        sources[rel] = sha(source)
        assert sha(dest) == sha(source)
        obj = dest.stem + ".o"
        subprocess.run([args.compiler, *flags, "-c", str(dest), "-o", obj],
                       cwd=args.build, check=True, timeout=30)
        objects.append(obj)
    driver = Path(__file__).with_name("operators_oracle.f90")
    subprocess.run([args.compiler, *flags, str(driver), *objects, "-o", "operators_oracle"],
                   cwd=args.build, check=True, timeout=30)
    records = cases()
    txt = io.StringIO()
    txt.write(f"{len(records)}\n")
    for rec in records:
        txt.write(f"{rec['n']} {rec['dt']} {rec['roughness']} {rec['ustar']} {rec['phi_epsilon']} {rec['phi_m']}\n")
        for name in FIELDS:
            a = np.asarray(rec["input"][name], np.float32)
            if len(a) == rec["n"]:
                a = np.r_[a, a[-1]]
            txt.write(" ".join(f"{v:.9e}" for v in a)+"\n")
    result = subprocess.run([str(args.build / "operators_oracle")], input=txt.getvalue(),
                            cwd=args.build, capture_output=True, text=True, check=True, timeout=30)
    (args.build / "input.txt").write_text(txt.getvalue())
    (args.build / "output.txt").write_text(result.stdout)
    lines = iter(result.stdout.splitlines())
    for rec in records:
        rec["output"] = {}
        for _ in range(18):
            name, values = next(lines), [float(v) for v in next(lines).split()]
            assert np.isfinite(values).all(), (rec["name"], name, values)
            rec["output"][name] = values
    assert next(lines, None) is None
    fixture = {"scope": "WRF 4.7.1 deterministic subroutines ONLY; no full PBL/coupled qualification",
               "precision": "WRF REAL32", "levels": "KEPS bottom-up; EEPS solver preserves caller order",
               "source_sha256": sources, "driver_sha256": sha(driver), "compiler_flags": flags,
               "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
               "input_sha256": hashlib.sha256(txt.getvalue().encode()).hexdigest(),
               "tolerance": {"field_scale_float32_ulps": 16, "zero_fields": "exact", "finite": "both arms required"},
               "records": records}
    args.output.write_text(json.dumps(fixture, indent=1)+"\n")
    print(json.dumps({"records": len(records), "outputs_per_record": len(records[0]["output"]),
                      "fixture_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
