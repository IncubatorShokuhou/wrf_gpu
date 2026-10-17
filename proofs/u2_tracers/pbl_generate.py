"""Extract and execute pristine WRF diff4d/diff/invert; imports no JAX.

Run under the lane's QUIET guard, CPU affinity, nice19 and timeout. Build files
stay in --build-dir and may be removed after the generated fixture is checked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def input_hashes(payload):
    """Seal REAL32 caller operands, including shape and field identity."""
    records = {}
    for name, record in payload.items():
        digest = hashlib.sha256()
        for key in ("q", "rho", "dz", "exch_h", "exch_m", "dt"):
            value = np.asarray(record[key], dtype="<f4")
            digest.update(json.dumps([key, list(value.shape)]).encode())
            digest.update(value.tobytes())
        records[name] = digest.hexdigest()
    return records, sha(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())


def extract(source):
    text = source.read_text()
    start = text.index("          SUBROUTINE diff4d(")
    stop = text.index("      END SUBROUTINE diff4d", start) + len("      END SUBROUTINE diff4d")
    first = text[start:stop]
    start2 = text.index("       subroutine diff(", stop)
    stop2 = text.index("        end subroutine invert", start2) + len("        end subroutine invert")
    return first + "\n" + text[start2:stop2]


def cases():
    rng = np.random.default_rng(20261013)
    for name, nz, dt in (("heterogeneous", 8, 54), ("strong_exchange", 8, 180),
                         ("zero_exchange", 8, 18), ("constant_profile", 8, 54),
                         ("one_level", 1, 54), ("two_levels", 2, 18)):
        q = rng.uniform(0.001, 3, (8, nz, 3, 4)).astype("f4")
        rho = rng.uniform(0.25, 1.2, (nz, 3, 4)).astype("f4")
        dz = rng.uniform(20, 200, (nz, 3, 4)).astype("f4")
        kh = rng.uniform(0.1, 300, (nz + 1, 3, 4)).astype("f4")
        km = (kh * np.float32(1.7)).astype("f4")
        if name == "strong_exchange":
            kh *= np.float32(100)
        if name == "zero_exchange":
            kh.fill(0)
        if name == "constant_profile":
            q[:] = q[:, :1]
        yield name, q, rho, dz, kh, km, np.float32(dt)


DRIVER = """
program oracle
use pbl_extracted
implicit none
integer :: n,nz,ny,nx
real :: dt
real,allocatable :: q(:,:,:,:),rho(:,:,:),dz(:,:,:),kh(:,:,:),km(:,:,:),out(:,:,:,:)
open(11,file='input.bin',access='stream',form='unformatted',status='old')
read(11) n,nz,ny,nx,dt
allocate(q(nx,nz+1,ny,n),rho(nx,nz+1,ny),dz(nx,nz+1,ny),kh(nx,nz+1,ny),km(nx,nz+1,ny),out(nx,nz+1,ny,n))
q=0.
rho=1.
dz=1.
read(11) q(:,1:nz,:,:),rho(:,1:nz,:),dz(:,1:nz,:),kh,km
close(11)
out=-777.
! All 8 tracer lanes active, no dummy scalar slot needed in this extraction.
call diff4d(dt,dz,q,.false.,rho,kh,km,out,n,1, &
  1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz)
open(12,file='output.bin',access='stream',form='unformatted',status='replace')
write(12) out(:,1:nz,:,:)
close(12)
end program oracle
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_pbl_driver.F"))
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("literal", "fma"), default="literal")
    p.add_argument("--init", choices=("zero", "snan"), default="zero")
    p.add_argument("--compare", type=Path, help="same-input reference fixture")
    p.add_argument("--receipt", type=Path, help="write finite/byte/spread comparison receipt")
    args = p.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    routines = extract(args.source)
    stub = "module module_state_description\ninteger,parameter :: P_QNS=91,P_QNR=92,P_QNG=93,P_QT=94,P_QNH=95,P_QVOLG=96,P_QKE_ADV=97\nend module\n"
    generated = stub + "module pbl_extracted\ncontains\n" + routines + "\nend module\n" + DRIVER
    (build / "oracle.f90").write_text(generated)
    arithmetic_flags = (["-O0", "-ffp-contract=off"] if args.mode == "literal"
                        else ["-O2", "-ffp-contract=fast", "-march=native", "-mfma"])
    flags = arithmetic_flags + ["-fcheck=all", "-ffree-line-length-none",
                                "-finit-real=" + args.init, "-finit-integer=0",
                                "-finit-logical=false"]
    compiler = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")
    started = time.perf_counter()
    subprocess.run([compiler, *flags, "oracle.f90", "-o", "oracle"], cwd=build, check=True)
    compile_seconds = time.perf_counter() - started
    payload = {}
    for name, q, rho, dz, kh, km, dt in cases():
        n, nz, ny, nx = q.shape
        with (build / "input.bin").open("wb") as f:
            np.array([n,nz,ny,nx], dtype="i4").tofile(f)
            np.array(dt, dtype="f4").tofile(f)
            # Python [N,z,y,x] -> Fortran [x,z,y,N], with x contiguous.
            q.transpose(0,2,1,3).tofile(f)
            for a in (rho,dz,kh,km):
                a.transpose(1,0,2).tofile(f)
        subprocess.run([str(build / "oracle")], cwd=build, check=True)
        out = np.fromfile(build / "output.bin", dtype="f4").reshape(n,ny,nz,nx).transpose(0,2,1,3)
        assert np.isfinite(out).all()
        payload[name] = {k: v.tolist() for k,v in dict(q=q,rho=rho,dz=dz,exch_h=kh,exch_m=km,tendency=out,dt=dt).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    comparison = None
    if args.compare:
        reference = json.loads(args.compare.read_text())
        assert set(reference) == set(payload), "reference cases differ"
        records = []
        for name, result in payload.items():
            original = reference[name]
            input_bytes_equal = all(
                np.array_equal(np.asarray(result[k], dtype="f4").view("u4"),
                               np.asarray(original[k], dtype="f4").view("u4"))
                for k in ("q", "rho", "dz", "exch_h", "exch_m", "dt"))
            assert input_bytes_equal, "comparison requires byte-identical caller inputs"
            actual = np.asarray(result["tendency"], dtype="f4")
            expected = np.asarray(original["tendency"], dtype="f4")
            different = actual.view("u4") != expected.view("u4")
            difference = np.abs(actual.astype("f8") - expected.astype("f8"))
            fixed = 16 * np.finfo(np.float32).eps * (
                np.abs(np.asarray(original["q"], dtype="f4")) / np.float32(original["dt"])
                + np.abs(expected))
            records.append({"case": name, "input_byte_equal": input_bytes_equal,
                            "output_values": int(actual.size),
                            "output_nonbyte": int(different.sum()),
                            "finite": bool(np.isfinite(actual).all()),
                            "max_abs_spread": float(difference.max()),
                            "fixed_literal_bound_excess_count": int((difference > fixed).sum()),
                            "max_spread_over_fixed_literal_bound": float((difference / fixed).max())})
        comparison = {"reference": str(args.compare),
                      "reference_sha256": sha(args.compare.read_bytes()),
                      "all_inputs_byte_equal": True,
                      "all_outputs_byte_equal": all(r["output_nonbyte"] == 0 for r in records),
                      "all_outputs_finite": all(r["finite"] for r in records),
                      "records": records}
    case_inputs, combined_inputs = input_hashes(payload)
    manifest = {
        "source": str(args.source), "source_sha256": sha(args.source.read_bytes()),
        "extracted_routines_sha256": sha(routines.encode()),
        "generated_fortran_sha256": sha(generated.encode()),
        "fixture_sha256": sha(args.output.read_bytes()),
        "input_sha256": combined_inputs, "case_input_sha256": case_inputs,
        "routines": "diff4d 2690:2818, diff 2822:2896, invert 2899:2931",
        "precision": "Fortran default REAL32; gfortran " + " ".join(flags),
        "mode": args.mode, "initialization": args.init, "fp_traps": False,
        "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "compiler_flags": flags,
        "generator": "proofs/u2_tracers/pbl_generate.py",
        "generation_command": ["python", "proofs/u2_tracers/pbl_generate.py",
                               "--source", str(args.source), "--build-dir", str(args.build_dir),
                               "--output", str(args.output), "--mode", args.mode, "--init", args.init]
                              + (["--compare", str(args.compare)] if args.compare else []),
        "variables": {"q": "[N,z,y,x], dimensionless mixing ratio", "rho": "[z,y,x], kg/m3", "dz": "[z,y,x], m", "exch_h/m": "[z+1,y,x], m2/s", "dt": "scalar, PBL DTBL seconds", "tendency": "[N,z,y,x], mixing ratio/s"},
        "boundary": "bottom exchange zero, top mass-level concentration held fixed",
        "tolerance": "predeclared: abs(error) <= 16*eps32*(abs(q)/dt + abs(reference tendency)); no value-dependent recalibration",
        "cases": list(payload), "seed": 20261013,
        "comparison": comparison,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps({"mode": args.mode, "init": args.init,
                                           "compile_seconds": compile_seconds,
                                           "fixture_sha256": manifest["fixture_sha256"],
                                           "source_sha256": manifest["source_sha256"],
                                           "generated_fortran_sha256": manifest["generated_fortran_sha256"],
                                           "input_sha256": combined_inputs,
                                           "case_input_sha256": case_inputs,
                                           "extracted_routines_sha256": manifest["extracted_routines_sha256"],
                                           "comparison": comparison}, indent=2) + "\n")
    print(json.dumps({"cases": len(payload), "fixture": str(args.output), "sha256": manifest["fixture_sha256"]}))


if __name__ == "__main__":
    main()
