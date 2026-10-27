"""Isolated changed-Nt_c calc_effectRad oracle on unchanged history inputs.

Reuse one build_oracle.py private module via --module-dir; never rebuild or
modify the original WRF module. Run separately for Nt_c=50/100/150e6. The
wrapper initializes WRF, calls ORIGINAL calc_effectRad, and applies the
ORIGINAL mp_gt_driver :1475–1477 writeback bounds. No microphysics step runs.

Bounds frozen before this tool's first measurement: cloud/ice <=8 REAL32 ULP;
snow <=1e-5 relative, no absolute floor. The historical X2 producer study
(b-thompson/X2/re/re_score.json) found cloud0ULP/ice2ULP/snow6.783e-6 relative
(94ULP) and counted cells above1e-5. Eight ULP covers the short density/pow/
DOUBLE-divide chain; snow retains the established polynomial-order allowance.
These are radius-operator bounds, not radiation flux or forecast tolerances.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


BOUNDS = {"cloud": {"real_ulp": 8}, "ice": {"real_ulp": 8}, "snow": {"relative": 1e-5}}
VALUES = (50.e6, 100.e6, 150.e6)
FORTRAN = """PROGRAM ntc_radius_oracle
  USE module_mp_thompson, ONLY: thompson_init, calc_effectRad
  USE module_model_constants, ONLY: RE_QC_BG, RE_QI_BG, RE_QS_BG
  IMPLICIT NONE
  INTEGER :: ncol,nlev,i,k
  REAL, ALLOCATABLE :: th(:,:),pii(:,:),p(:,:),qv(:,:),qc(:,:),qi(:,:),ni(:,:),qs(:,:),dz(:,:)
  REAL, ALLOCATABLE :: t(:,:),rec(:,:),rei(:,:),res(:,:),nc_zero(:),hgt(:,:,:)
  OPEN(10,FILE='input.bin',ACCESS='stream',FORM='unformatted',STATUS='old')
  READ(10) ncol,nlev
  ALLOCATE(th(ncol,nlev),pii(ncol,nlev),p(ncol,nlev),qv(ncol,nlev),qc(ncol,nlev))
  ALLOCATE(qi(ncol,nlev),ni(ncol,nlev),qs(ncol,nlev),dz(ncol,nlev),t(ncol,nlev))
  ALLOCATE(rec(ncol,nlev),rei(ncol,nlev),res(ncol,nlev),nc_zero(nlev),hgt(ncol,nlev,1))
  READ(10) th,pii,p,qv,qc,qi,ni,qs,dz
  CLOSE(10)
  hgt(:,1,1)=0.
  DO k=2,nlev
    hgt(:,k,1)=hgt(:,k-1,1)+dz(:,k-1)
  END DO
  CALL thompson_init(HGT=hgt, IDS=1,IDE=ncol+1,JDS=1,JDE=2,KDS=1,KDE=nlev+1, &
       IMS=1,IME=ncol,JMS=1,JME=1,KMS=1,KME=nlev,ITS=1,ITE=ncol,JTS=1,JTE=1,KTS=1,KTE=nlev)
  t=th*pii
  nc_zero=0. ! mp8 non-aerosol calc_effectRad ignores nc1d and uses its compiled Nt_c.
  DO i=1,ncol
    CALL calc_effectRad(t(i,:),p(i,:),qv(i,:),qc(i,:),nc_zero,qi(i,:),ni(i,:),qs(i,:), &
                        rec(i,:),rei(i,:),res(i,:),1,nlev)
    rec(i,:)=MAX(RE_QC_BG,MIN(rec(i,:),50.E-6))
    rei(i,:)=MAX(RE_QI_BG,MIN(rei(i,:),125.E-6))
    res(i,:)=MAX(RE_QS_BG,MIN(res(i,:),999.E-6))
  END DO
  OPEN(11,FILE='output.bin',ACCESS='stream',FORM='unformatted',STATUS='new')
  WRITE(11) t,rec,rei,res
  CLOSE(11)
END PROGRAM ntc_radius_oracle
"""


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, reference):
    rows = {}
    for name in BOUNDS:
        a, b = np.asarray(actual[name]), np.asarray(reference[name])
        assert a.shape == b.shape and a.dtype == b.dtype == np.float32, name
        assert np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all(), name
        error = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ulp = np.abs(a.view(np.uint32).astype(np.int64) - b.view(np.uint32).astype(np.int64))
        relative = error / np.abs(b.astype(np.float64))
        ratio = ulp / BOUNDS[name]["real_ulp"] if "real_ulp" in BOUNDS[name] else relative / BOUNDS[name]["relative"]
        rows[name] = dict(max_abs=float(error.max()), max_relative=float(relative.max()), max_real_ulp=int(ulp.max()),
                          failed_cells=int(np.count_nonzero(ratio > 1)), max_bound_ratio=float(ratio.max()),
                          byte_exact=a.tobytes() == b.tobytes())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--nt-c", type=float, choices=VALUES, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("module_dir", "fixture", "selection", "source_root", "out"):
        setattr(args, name, getattr(args, name).resolve())
    assert os.environ.get("JAX_PLATFORMS") == "cpu", "CPU oracle only; set JAX_PLATFORMS before imports"
    assert os.environ.get("GPUWRF_JAX_CACHE") == "0"
    pin = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    assert subprocess.run(["git", "-C", str(args.source_root), "symbolic-ref", "--quiet", "HEAD"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 1, "measure a detached source snapshot"
    assert not subprocess.check_output(["git", "-C", str(args.source_root), "status", "--porcelain", "--untracked-files=no"]), "dirty source"
    assert not args.out.exists(), "refuse to overwrite an arm"
    module_receipt = json.loads((args.module_dir / "receipt.json").read_text())
    assert set(module_receipt["values"]) <= {"nt_c"}, "only Nt_c may differ in this original module"
    assert np.float32(module_receipt["values"].get("nt_c", 100.e6)) == np.float32(args.nt_c)
    assert sha(args.module_dir / "module_mp_thompson.o") == module_receipt["object_sha256"]
    assert sha(args.module_dir / "module_mp_thompson.f90") == module_receipt["changed_f90_sha256"]
    assert (args.module_dir / "module_mp_thompson.mod").is_file()
    literal = re.search(r"REAL,\s*PARAMETER,\s*PRIVATE::\s*Nt_c\s*=\s*([\d.EeDd+\-]+)",
                        (args.module_dir / "module_mp_thompson.f90").read_text(), re.I)[1]
    assert np.float32(float(literal.replace("D", "E").replace("d", "e"))) == np.float32(args.nt_c)
    selection = json.loads(args.selection.read_text())["selected_columns"]
    assert len(selection) == len(set(selection)) == 24
    assert module_receipt["selection"] == selection and module_receipt["fixture_sha256"] == sha(args.fixture)
    z = dict(np.load(args.fixture, allow_pickle=False))
    assert all(np.isfinite(value).all() for name,value in z.items() if name.startswith(("in_", "wrf_")))
    idx = np.array([z["names"].tolist().index(name) for name in selection])
    assert np.all(z["dt"][idx] == 18.)
    ncol, nlev = len(idx), z["in_qv"].shape[1]
    assert all(np.count_nonzero(z["in_" + name][idx]) for name in ("qc", "qi", "qs"))
    args.out.mkdir(parents=True)
    (args.out / "GATE.json").write_text(json.dumps(dict(bounds=BOUNDS, units="m", chosen_before_measurement=True,
        basis="Historical X2 radius study; short REAL chain <=8 ULP, existing snow polynomial criterion <=1e-5 relative"), indent=2) + "\n")
    (args.out / "driver.f90").write_text(FORTRAN)
    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    buildenv = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build")
    compiler = buildenv / "bin/x86_64-conda-linux-gnu-gfortran"
    flags = ["-w", "-ffree-form", "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4",
             "-fallow-argument-mismatch", "-fallow-invalid-boz", "-O2", "-ftree-vectorize", "-funroll-loops"]
    tables = {}
    for name, row in module_receipt["tables"].items():
        target = args.module_dir / name
        assert target.is_file() and target.samefile(row["path"]), name
        (args.out / name).symlink_to(target.resolve())
        tables[name] = dict(path=str(target.resolve()), sha256_reused_from_module_receipt=row["sha256"])
    with (args.out / "input.bin").open("wb") as stream:
        stream.write(np.asarray([ncol, nlev], ">i4").tobytes())
        for name in ("th", "pii", "p", "qv", "qc", "qi", "ni", "qs", "dz8w"):
            stream.write(np.asarray(z["in_" + name][idx].T, ">f4").tobytes())
    receipt = dict(scope="Original calc_effectRad on unchanged pre-MP history inputs; not radiation/forecast fidelity",
                   platform="cpu", gpu_executed=False, nt_c=args.nt_c, bounds=BOUNDS, selected_columns=selection,
                   fixture_sha256=sha(args.fixture), selection_sha256=sha(args.selection),
                   original_module_receipt_sha256=sha(args.module_dir / "receipt.json"),
                   original_module_object_sha256=module_receipt["object_sha256"],
                   original_module_source_sha256=module_receipt["changed_f90_sha256"],
                   driver_sha256=sha(args.out / "driver.f90"), tool_sha256=sha(Path(__file__)), tables=tables,
                   source_pin=pin, source_root=str(args.source_root),
                   work=dict(ncol=ncol, nlev=nlev, nc1d_input="zero; ignored by original mp8", caller_clamps="Thompson.F:1475–1477"))
    commands = []

    def run(command):
        commands.append([str(part) for part in command])
        with (args.out / "build.log").open("a") as log:
            subprocess.run(commands[-1], cwd=args.out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)

    started = time.monotonic()
    try:
        inc = ["-I" + str(args.module_dir)] + ["-I" + str(wrf / name) for name in ("phys", "frame", "share", "main", "inc")]
        run([compiler, *flags, *inc, "-c", "driver.f90", "-o", "driver.o"])
        run([compiler, *flags, "driver.o", args.module_dir / "module_mp_thompson.o", wrf / "main/libwrflib.a",
             wrf / "external/fftpack/fftpack5/libfftpack.a", wrf / "external/io_grib1/libio_grib1.a",
             wrf / "external/io_grib_share/libio_grib_share.a", wrf / "external/io_int/libwrfio_int.a",
             "-L" + str(wrf / "external/esmf_time_f90"), "-lesmf_time", wrf / "frame/module_internal_header_util.o",
             wrf / "frame/pack_utils.o", "-L" + str(wrf / "external/io_netcdf"), "-lwrfio_nf",
             "-L" + str(buildenv / "lib"), "-lnetcdff", "-lnetcdf", "-Wl,-rpath," + str(buildenv / "lib"), "-o", "oracle"])
        receipt["executable_sha256"] = sha(args.out / "oracle")
        run([args.out / "oracle"])
        raw = np.fromfile(args.out / "output.bin", ">f4")
        assert raw.size == 4 * ncol * nlev and np.isfinite(raw).all()
        reference = {name: part.reshape(nlev, ncol).T.astype(np.float32)
                     for name, part in zip(("T", "cloud", "ice", "snow"), np.split(raw, 4), strict=True)}
        assert reference["T"].tobytes() == (z["in_th"][idx] * z["in_pii"][idx]).astype(np.float32).tobytes()
        np.savez_compressed(args.out / "reference.npz", **reference)
        sys.path.insert(0, str(args.source_root / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf._fast_defaults import apply_fast_path_defaults
        apply_fast_path_defaults()
        from gpuwrf.physics.thompson_nt_c import effective_radii
        from gpuwrf.physics.rrtmg_mp_re import calc_thompson_effective_radii
        assert jax.devices()[0].platform == "cpu"
        Inputs = namedtuple("Inputs", "T p qv qc qi Ni qs")
        state = Inputs(jnp.asarray(reference["T"]), *(jnp.asarray(z["in_" + name][idx], jnp.float32)
                                                   for name in ("p", "qv", "qc", "qi", "ni", "qs")))
        traces = []

        def controlled(state, nt_c):
            traces.append(1)
            return effective_radii(state, nt_c)

        compiled = jax.jit(controlled)
        ports = {}
        for value in VALUES:
            radii = compiled(state, jnp.float32(value))
            ports[value] = {name: np.asarray(array) for name, array in zip(BOUNDS, radii, strict=True)}
            assert all(np.isfinite(array).all() for array in ports[value].values())
            np.savez_compressed(args.out / f"port_{int(value)}.npz", **ports[value])
        assert len(traces) == 1, traces
        static = jax.jit(lambda s: calc_thompson_effective_radii(*s))(state)
        static = {name: np.asarray(array) for name, array in zip(BOUNDS, static, strict=True)}
        receipt["default_vs_static_producer"] = compare(ports[100.e6], static)
        receipt["candidate"] = compare(ports[args.nt_c], reference)
        receipt["default_omission_control"] = compare(ports[100.e6], reference)
        receipt["changed_cloud_cells"] = {str(int(value)): int(np.count_nonzero(ports[value]["cloud"] != ports[100.e6]["cloud"]))
                                           for value in (50.e6, 150.e6)}
        assert all(receipt["changed_cloud_cells"].values())
        for value in VALUES:
            for name in ("ice", "snow"):
                assert ports[value][name].tobytes() == ports[100.e6][name].tobytes(), (value, name)
        receipt.update(jit_traces=len(traces), candidate_pass=all(row["failed_cells"] == 0 for row in receipt["candidate"].values()),
                       omission_killed=None if args.nt_c == 100.e6 else receipt["default_omission_control"]["cloud"]["failed_cells"] > 0,
                       radius_source_sha256=sha(args.source_root / "src/gpuwrf/physics/thompson_nt_c.py"))
        assert subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip() == pin
        assert receipt["candidate_pass"], receipt["candidate"]
        assert args.nt_c == 100.e6 or receipt["omission_killed"], "default-Nt omission control survived"
        receipt["status"] = "PASS"
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(commands=commands, wall_s=time.monotonic() - started)
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for name in ("oracle", "driver.o", "input.bin", "output.bin"):
            (args.out / name).unlink(missing_ok=True)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
