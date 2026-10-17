"""Generate small pristine-WRF KF-CuP component fixtures (CPU only).

Usage: python generate.py --build-dir <lane disk dir> --output <fixtures.json>
Each exact original procedure is extracted by a bounded start/end match.
The entire original module_model_constants is compiled unchanged (E83).
No WRF full-model executable is used, so there is no E104 shared dump path.
Run only through a timeout, nice 19, taskset and the repository QUIET guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PROCEDURES = ("cupCloudFraction", "cup_jfd", "findCp", "findRs", "findRsi")


def radiation_block(source):
    lines = source.splitlines(keepends=True)
    start = [i for i,l in enumerate(lines) if l.strip() == "!Modify CLDFRA and QC for kfcupscheme cumulus scheme"]
    if len(start) != 1:
        raise ValueError("ambiguous pristine KF-CuP radiation block")
    # The first block is exactly delimited by its final three endifs. A
    # bounded IF depth parser ignores comments and cannot swallow another
    # scheme after a changed source layout.
    depth, end = 0, None
    for i in range(start[0]+1,min(start[0]+100,len(lines))):
        code = lines[i].split("!")[0].strip().lower()
        if re.match(r"if\s*\(.*\)\s*then\s*$",code):
            depth += 1
        elif re.match(r"end\s*if\b|endif\b",code):
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None or depth != 0:
        raise ValueError("unbalanced pristine KF-CuP radiation block")
    return "".join(lines[start[0]:end+1]), [start[0]+1,end+1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(source, name):
    kind = "function" if name.startswith("find") else "subroutine"
    lines = source.splitlines(keepends=True)
    start = [i for i, line in enumerate(lines) if re.match(rf"\s*{kind}\s+{name}\s*\(", line, re.I)]
    end = [i for i, line in enumerate(lines) if re.match(rf"\s*end\s+{kind}\s+{name}\s*$", line, re.I)]
    if len(start) != 1 or len(end) != 1 or end[0] < start[0]:
        raise ValueError(f"ambiguous or missing pristine procedure: {name}")
    return "".join(lines[start[0]:end[0] + 1]), [start[0] + 1, end[0] + 1]


def cases():
    out = []
    for name, scalars, n in (
        ("normal", [1200., -800., .002, .8, .1, .0001], 9),
        ("even_bins", [1200., -800., .002, .8, .1, .0001], 8),
        ("negative_sigmas", [-500., 800., -.002, -.8, .15, .0002], 9),
        ("positive_slope_floor", [1200., 1., .002, .8, .1, .0001], 9),
        ("negative_slope_floor", [1200., -1., .002, .8, .1, .0001], 9),
        ("slope_upper_cap", [1200., 3000., .002, .8, .1, .0001], 9),
        ("slope_lower_cap", [1200., -3000., .002, .8, .1, .0001], 9),
        ("equal_slopes", [800., 800., .002, .8, .1, .0001], 9),
        ("zero_parameters", [0., 0., 0., 0., .1, .0001], 1),
        ("one_bin", [800., -800., .002, .8, .1, .0001], 1),
        ("boundary_slopes", [2000., 10., .002, .8, .1, .0001], 9),
    ):
        out.append({"id": "jfd_" + name, "routine": "cup_jfd", "num_bins": n,
                    "scalars": scalars})

    # Inputs at the component boundary, not a claim of full convection-driver
    # activation: ISHALL and parcel mass flux come from the unported driver.
    base = dict(qlg=[0, .0005, .001, .0015, .0002, 0], qig=[0, 0, .0001, .0002, .0001, 0],
                qv=[.014, .011, .007, .004, .002, .001], temperature=[298, 292, 284, 275, 265, 250],
                height=[100, 400, 900, 1600, 2400, 3300], pressure=[99000, 95000, 89000, 82000, 73000, 64000],
                mfup=[0, .0002, .001, .003, .0005, 0])
    variations = (
        ("absent", dict(ishall=2)), ("deep", dict(ishall=0)),
        ("deep_zero_flux", dict(ishall=0, mfup=[0]*6)),
        ("deep_upper_cap", dict(ishall=0, mfup=[0, 100, 100, 100, 100, 0])),
        ("shallow", {}), ("shallow_wparcel_zero", dict(wparcel=0)),
        ("shallow_wparcel_negative", dict(wparcel=-1)),
        ("shallow_lower_cap", dict(active_frac=0)),
        ("shallow_upper_cap", dict(active_frac=2)),
        ("shallow_tau_cap", dict(wstar=.1)),
        ("shallow_lnint_floor", dict(qlg=[0]*6)),
        ("shallow_saturated", dict(qv=[.04]*6)),
        ("shallow_ice_only", dict(qlg=[0]*6, qig=[.001]*6)),
        ("shallow_top_boundary", dict(cloud_top=4)),
        ("shallow_negative_wstar", dict(wstar=-2, wparcel=-1)),
    )
    for name, edits in variations:
        item = dict(base, id="cloud_" + name, routine="cupCloudFraction", ishall=1,
                    cloud_bottom=1, cloud_top=3, wstar=2., wparcel=3., pblh=1000., active_frac=.1)
        item.update(edits)
        out.append(item)
    out.append({"id": "thermodynamics", "routine": "thermodynamics",
                "temperature": [220, 240, 260, 273.16, 290, 305],
                "pressure": [25000, 35000, 55000, 75000, 95000, 101000],
                "qv": [0, .0001, .001, .005, .015, .025]})
    for ishall in (0,1,2):
        for icloud in (0,1,3):
            for forced in (False,True):
                out.append(dict(id=f"radiation_s{ishall}_i{icloud}_f{int(forced)}", routine="cup_radiation",
                    ishall=ishall, icloud=icloud, shallowcu_forced_ra=forced, cloud_bottom=1., cloud_top=4.,
                    cldfra=[.2,.4,.6,.8,1.,.3], qc=[0.,0.,.002,.0001,0.,.0002],
                    cldfra_cup=[.1,.2,.3,.4,.5,.6], cldfra1_flag=[1,2,3,1,2,3]))
    return out


def line(values):
    return " ".join(str(float(np.float32(x))) for x in values) + "\n"


def payload(case):
    if case["routine"] == "cup_jfd":
        return f"1 {case['num_bins']}\n" + line(case["scalars"])
    if case["routine"] == "cup_radiation":
        n=len(case["cldfra"])
        return (f"4 {n}\n{case['ishall']} {case['cloud_bottom']+1} {case['cloud_top']+1} {case['icloud']} {int(case['shallowcu_forced_ra'])}\n"
                + "".join(line(case[k]) for k in ("cldfra","qc","cldfra_cup"))
                + " ".join(str(x) for x in case["cldfra1_flag"])+"\n")
    n = len(case["temperature"])
    if case["routine"] == "thermodynamics":
        return f"3 {n}\n" + "".join(line(case[k]) for k in ("temperature", "pressure", "qv"))
    text = f"2 {n}\n{case['ishall']} {case['cloud_bottom']+1} {case['cloud_top']+1}\n"
    text += line([case[k] for k in ("wstar", "wparcel", "pblh", "active_frac")] + [18.])
    return text + "".join(line(case[k]) for k in ("qlg", "qig", "qv", "temperature", "height", "pressure", "mfup"))


def decode(case, stdout):
    values = np.array([float(x) for x in stdout.split()], np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"pristine nonfinite: {case['id']}")
    if case["routine"] == "cup_jfd":
        n = case["num_bins"]
        assert len(values) == 4 + 2*n + n*n
        out = dict(zip(("slope_sfc", "slope_ez", "sigma_sfc", "sigma_ez"), values[:4].tolist()))
        out.update(th_perturb=values[4:4+n].tolist(), r_perturb=values[4+n:4+2*n].tolist(),
                   jfd=values[4+2*n:].reshape((n,n), order="F").tolist())
        return out
    if case["routine"] == "cup_radiation":
        n=len(case["cldfra"])
        assert len(values)==3*n
        return dict(zip(("cldfra","qc","cldfra_cup"),values.reshape(3,n).tolist()))
    n = len(case["temperature"])
    if case["routine"] == "thermodynamics":
        assert len(values) == 3*n
        return dict(zip(("cp", "rs", "rsi"), values.reshape(n, 3).T.tolist()))
    out = {"cldfra": values[:n].tolist()}
    expected = n
    if case["ishall"] != 2:
        out["tstar"] = float(values[n])
        expected += 1
    if case["ishall"] == 1:
        out.update(zip(("taucloud", "tactive", "lnint"), values[n+1:n+4].tolist()))
        out["lnterms"] = values[n+4:].tolist()
        expected += 3+n
    assert len(values) == expected
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("explicit JAX_PLATFORMS=cpu required")
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise RuntimeError("QUIET")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source_path = args.wrf_root / "phys/module_cu_kfcup.F"
    constants = args.wrf_root / "share/module_model_constants.F"
    radiation = args.wrf_root / "phys/module_radiation_driver.F"
    source = source_path.read_text()
    extracted = {name: extract(source, name) for name in PROCEDURES}
    original = "module kfcup_oracle_components\n use module_wrf_error\n implicit none\n real,parameter::eps=.622\ncontains\n"
    original += "\n".join(text for text, _ in extracted.values())
    block, block_bounds = radiation_block(radiation.read_text())
    # Wrapper declarations only; original numerical/control-flow block is
    # retained verbatim, including PRESENT and the Fortran bounds/tests.
    original += """
subroutine cup_radiation(its,ite,jts,jte,kts,kte,cu_physics,icloud,shallowcu_forced_ra,shall,cubot,cutop,cldfra1_flag,cldfra,qc,cldfra_cup)
 integer,intent(in)::its,ite,jts,jte,kts,kte,cu_physics,icloud
 integer,parameter::KFCUPSCHEME=10
 logical,intent(in)::shallowcu_forced_ra
 real,intent(in)::shall(its:ite,jts:jte),cubot(its:ite,jts:jte),cutop(its:ite,jts:jte)
 integer,intent(in)::cldfra1_flag(its:ite,kts:kte,jts:jte)
 real,intent(inout)::cldfra(its:ite,kts:kte,jts:jte),qc(its:ite,kts:kte,jts:jte)
 real,intent(inout),optional::cldfra_cup(its:ite,kts:kte,jts:jte)
 logical::compute_cldfra_cup
 real::cldfra_cup_mod
 integer::i,j,k
""" + block + "\nend subroutine cup_radiation\n"
    original += "end module kfcup_oracle_components\n"
    (build / "components.F90").write_text(original)
    shutil.copyfile(constants, build / "module_model_constants.F")
    shutil.copyfile(ROOT / "proofs/p0_4/oracle/module_wrf_error.f90", build / "module_wrf_error.f90")
    shutil.copyfile(Path(__file__).with_name("oracle_components.F90"), build / "driver.F90")
    compiler = shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    common = [compiler, "-O0", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DEM_CORE=1"]
    compile_sources = ["module_wrf_error.f90", "module_model_constants.F", "components.F90", "driver.F90"]
    subprocess.run(common + compile_sources + ["-o", "oracle"], cwd=build, check=True, timeout=120)
    subprocess.run(common + ["-fcheck=all", "-ffpe-trap=invalid,zero,overflow"] + compile_sources + ["-o", "oracle_checked"], cwd=build, check=True, timeout=120)
    records = []
    for case in cases():
        outputs = []
        for executable in ("oracle", "oracle_checked"):
            result = subprocess.run([str(build / executable)], input=payload(case), text=True, capture_output=True, check=True, timeout=10)
            outputs.append(decode(case, result.stdout))
        if outputs[0] != outputs[1]:
            raise ValueError(f"instrumented oracle differs (E26): {case['id']}")
        records.append({**case, "output": outputs[0]})
    manifest = {
        "fixture_id": "v040-kfcup-components-real32", "source": "WRF 4.7.1 pristine exact procedures",
        "source_commit": subprocess.run(["git", "-C", str(args.wrf_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "source_sha256": {"phys/module_cu_kfcup.F": sha(source_path), "share/module_model_constants.F": sha(constants),
                          "phys/module_radiation_driver.F":sha(radiation)},
        "procedures": {name: {"lines": bounds, "sha256": hashlib.sha256(text.encode()).hexdigest()} for name, (text,bounds) in extracted.items()},
        "radiation_block": {"lines":block_bounds,"sha256":hashlib.sha256(block.encode()).hexdigest()},
        "caller": {"cloud": "KF_cup_CPS:1072-1077; p1d in Pa, bottom-up, mfup_cup passed directly; ISHALL supplied at component boundary",
                   "jfd": "KF_cup_CPS:631-633; theta first/moisture second; slopes and sigmas INOUT"},
        "scope": "Components only; constructed cloud-regime inputs do NOT qualify the unported convection driver or radiation mediation",
        "precision": "WRF default REAL32 (no DOUBLE in these procedures)",
        "compiler": subprocess.run([compiler, "--version"], capture_output=True,text=True,check=True).stdout.splitlines()[0],
        "flags": common[1:], "instrumentation": "-fcheck=all -ffpe-trap=invalid,zero,overflow; all output values byte-identical to uninstrumented",
        "tolerances": {"jfd": {"relative": 32*float(np.finfo(np.float32).eps), "absolute": 1e-12},
                       "cloud": {"relative": 64*float(np.finfo(np.float32).eps), "absolute": 1e-10},
                       "thermodynamics": {"relative": 32*float(np.finfo(np.float32).eps), "absolute": 1e-12},
                       "radiation": {"relative":0.,"absolute":0.}},
        "cases": records,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(records), "E26_identical": len(records), "fixture_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
