"""JAX-dark host builder for the standalone native GPU shadow oracle."""
import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
from oracle import build, load, oracle, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--case", type=Path, default=Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227"))
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    case = args.case
    lib, receipt = build(wrf, args.out)
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    lib.solar_probe.argtypes = [ctypes.c_int] * 2 + [arr] * 2 + [ctypes.c_float] * 4 + [arr] * 5
    rows, data = [], {}
    for d in ("d02", "d03"):
        f = load(case / f"wrfinput_{d}")
        ny, nx = f["HGT"].shape
        old = np.zeros_like(f["HGT"])
        for call, hour in enumerate((0., 7.5, 8., 18., 18.5, 24. + 7.5)):
            minute = hour * 60
            declin = np.zeros(1, np.float32)
            solar_out = [np.zeros((nx, ny), np.float32, order="F") for _ in range(4)]
            F = lambda a: np.asfortranarray(a.T)
            lib.solar_probe(nx, ny, F(f["XLAT"]), F(f["XLONG"]), np.float32(58 + minute / 1440),
                            minute, 0., 30., declin, *solar_out)
            for nested in (False, True):
                mask, height, _ = oracle(lib, f, minute, old if nested else None, declin_override=declin[0])
                key = f"{d}_{call}_{int(nested)}"
                for name, value in (("terrain", f["HGT"]), ("latitude", f["XLAT"]), ("longitude", f["XLONG"]),
                                    ("sina", f["SINALPHA"]), ("cosa", f["COSALPHA"]), ("old_height", old),
                                    ("declin", declin[0]), ("coszen", solar_out[2].T), ("hrang", solar_out[3].T),
                                    ("mask", mask), ("height", height)):
                    data[key + "_" + name] = np.asarray(value)
                rows.append(dict(key=key, domain=d, minute=minute, nested=nested, dx_m=f["dx"], dy_m=f["dy"]))
                if nested:
                    next_old = height
            old = next_old
    path = args.out / "fixture.npz"
    np.savez_compressed(path, **data)
    receipt.update(rows=rows, fixture=str(path), fixture_sha256=sha(path),
                   scope="24 real WN3 root/nested operator cases, explicit sequential retained-height history; not a full forecast",
                   inputs={d: {"path": str(case / f"wrfinput_{d}"), "sha256": sha(case / f"wrfinput_{d}")} for d in ("d02", "d03")})
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert "jax" not in sys.modules, "host fixture builder must stay JAX-dark"
    print(json.dumps(dict(cases=len(rows), fixture_sha256=sha(path))))


if __name__ == "__main__":
    main()
