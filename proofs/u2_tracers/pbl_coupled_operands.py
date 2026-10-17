"""Matched-operand tracer PBL diagnostic from original WRF history frames.

No JAX is imported. This runs unchanged pristine diff4d/diff/invert with the
previous frame's prognostics and the following frame's PBL coefficients.
AL and ALB are mandatory: history ALT alone can be stale after the last RK/MP
calc_p_rho_phi call and is not next step's rk_step_prep ALT.

Run only once the original case's IO-only reference-preservation gate passes,
under the lane QUIET guard, allowed CPU affinity, nice19, and timeout. Outputs
and --build-dir must be in the lane artifact directory, not the source tree.
"""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame(dataset, path):
    with dataset(path) as ds:
        lookup = {name.upper(): name for name in ds.variables}
        required = ("AL", "ALB", "PH", "PHB", "QVAPOR", "EXCH_H", "EXCH_M") + tuple(
            f"TR17_{n}" for n in range(1, 9))
        missing = set(required) - set(lookup)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}; history ALT is insufficient")
        result = {key: np.asarray(ds.variables[lookup[key]][0], dtype=np.float32)
                  for key in required}
        text = b"".join(np.asarray(ds.variables[lookup["TIMES"]][0]).tolist()).decode()
        result["time"] = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")
    if any(not np.isfinite(value).all() for key, value in result.items() if key != "time"):
        raise ValueError(f"{path.name}: nonfinite reference input")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("<USER_HOME>/wrf_gpu2_lanes/u2-tracers/TR17_coeff_case"))
    parser.add_argument("--source", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_pbl_driver.F"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("literal", "fma"), default="literal")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    generator = module_from_file("pbl_generate", Path(__file__).with_name("pbl_generate.py"))
    # Load the production NetCDF lock directly without importing gpuwrf/JAX.
    nc = module_from_file("pbl_diagnostic_netcdf_lock", repo / "src/gpuwrf/io/netcdf_lock.py")
    paths = sorted(args.case.glob("wrfout_d01_*"))
    if len(paths) != 4:
        raise ValueError("expected exactly four original frames for three dt18 steps")
    frames = [read_frame(nc.Dataset, path) for path in paths]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    routines = generator.extract(args.source)
    stub = "module module_state_description\ninteger,parameter :: P_QNS=91,P_QNR=92,P_QNG=93,P_QT=94,P_QNH=95,P_QVOLG=96,P_QKE_ADV=97\nend module\n"
    generated = stub + "module pbl_extracted\ncontains\n" + routines + "\nend module\n" + generator.DRIVER
    (build / "oracle.f90").write_text(generated)
    flags = (["-O0", "-ffp-contract=off"] if args.mode == "literal" else
             ["-O2", "-ffp-contract=fast", "-march=native", "-mfma"])
    flags += ["-fcheck=all", "-ffree-line-length-none", "-finit-real=zero",
              "-finit-integer=0", "-finit-logical=false"]
    compiler = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")
    started = time.perf_counter()
    subprocess.run([compiler, *flags, "oracle.f90", "-o", "oracle"], cwd=build, check=True)
    compile_seconds = time.perf_counter() - started
    gathered = {key: [] for key in ("tracers", "qv", "alt", "rho", "dz", "exch_h", "exch_m", "dt", "tendency")}
    records = []
    for step, (previous, current) in enumerate(zip(frames[:-1], frames[1:]), 1):
        seconds = (current["time"] - previous["time"]).total_seconds()
        if seconds != 18:
            raise ValueError("this source pairing is registered only for dt18, bldt0 three-step case")
        dt = np.float32(seconds)
        q = np.stack([previous[f"TR17_{n}"] for n in range(1, 9)])
        # rk_step_prep -> calc_alt (BSU:943), then phy_prep (BSU:4856).
        alt = np.add(previous["AL"], previous["ALB"], dtype=np.float32)
        rho = np.multiply(np.divide(np.float32(1), alt, dtype=np.float32),
                          np.add(np.float32(1), previous["QVAPOR"], dtype=np.float32), dtype=np.float32)
        # BSU:4869,4877: divide each interface total first, then subtract.
        zw = np.divide(np.add(previous["PHB"], previous["PH"], dtype=np.float32),
                       np.float32(9.81), dtype=np.float32)
        dz = np.subtract(zw[1:], zw[:-1], dtype=np.float32)
        kh, km = current["EXCH_H"], current["EXCH_M"]
        n, nz, ny, nx = q.shape
        if rho.shape != (nz, ny, nx) or dz.shape != rho.shape or kh.shape != (nz+1,ny,nx) or km.shape != kh.shape:
            raise ValueError("unexpected original WRF level/face shapes")
        if np.any(rho <= 0) or np.any(dz <= 0):
            raise ValueError("nonpositive original column coefficient")
        with (build / "input.bin").open("wb") as handle:
            np.asarray([n,nz,ny,nx], dtype="i4").tofile(handle)
            np.asarray(dt, dtype="f4").tofile(handle)
            q.transpose(0,2,1,3).tofile(handle)
            for array in (rho,dz,kh,km):
                array.transpose(1,0,2).tofile(handle)
        subprocess.run([str(build / "oracle")], cwd=build, check=True)
        tendency = np.fromfile(build / "output.bin", dtype="f4").reshape(n,ny,nz,nx).transpose(0,2,1,3)
        if not np.isfinite(tendency).all():
            raise ValueError("original extracted diff4d returned nonfinite tendency")
        operands = dict(q=q,rho=rho,dz=dz,exch_h=kh,exch_m=km,dt=dt)
        _, input_sha = generator.input_hashes({f"step{step}": operands})
        records.append({"step": step, "previous_frame": paths[step-1].name,
                        "coefficient_frame": paths[step].name, "input_sha256": input_sha,
                        "tendency_sha256": generator.sha(tendency.tobytes()),
                        "tendency_shape": list(tendency.shape), "nonfinite": 0,
                        "max_abs_tendency": float(np.max(np.abs(tendency))),
                        "top_tendency_nonzero": int(np.count_nonzero(tendency[:, -1]))})
        for key, value in dict(tracers=q,qv=previous["QVAPOR"],alt=alt,rho=rho,dz=dz,
                               exch_h=kh,exch_m=km,dt=dt,tendency=tendency).items():
            gathered[key].append(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{key: np.stack(value) for key,value in gathered.items()})
    receipt = {"scope": "matched-original-operands raw PBL diagnostic; not whole-trajectory truth",
               "mode": args.mode, "source_sha256": sha_file(args.source),
               "extracted_routines_sha256": generator.sha(routines.encode()),
               "generated_fortran_sha256": generator.sha(generated.encode()),
               "generator_sha256": sha_file(Path(__file__)),
               "frame_sha256": {path.name: sha_file(path) for path in paths},
               "compiler": compiler, "compiler_flags": flags, "fp_traps": False,
               "compile_seconds": compile_seconds, "records": records,
               "output_sha256": sha_file(args.output),
               "pairing": "previous AL+ALB/QV/TR17/PH/PHB, next EXCH_H/M; dt18 bldt0",
               "source_witnesses": ["solve_em.F:658 rk_step_prep; module_em.F:176 calc_alt",
                                    "module_big_step_utilities_em.F:943,4856,4869,4877",
                                    "module_first_rk_step_part1.F:208,1179,1307",
                                    "module_pbl_driver.F:2353-2362,2779-2806",
                                    "module_bl_mynnedmf_driver.F:596-597"],
               "bottom_top": "bottom heat exchange overwritten0; face index nz ignored; top mass row identity",
               "gate": "diagnostic only; fixed literal16eps and coupled1e-4 unchanged"}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
