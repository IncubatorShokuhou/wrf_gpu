"""Build original WRF mp38 with BOTH production init switches enabled.

Unlike v018's hail-only oracle this uses physics_init.F:4565's aerosol
arguments as well as NG. Original scheme files are copied verbatim. Only
the project-authored caller and production-control stubs are authored here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main(wrf: Path, out: Path):
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    old = repo / "proofs/v018/mp_oracles/thompgh/oracle"
    out.mkdir(parents=True, exist_ok=True)
    sources = {"module_model_constants.F": wrf / "share/module_model_constants.F",
               "libmassv.F": wrf / "frame/libmassv.F",
               "module_mp_radar.F": wrf / "phys/module_mp_radar.F",
               "module_mp_thompson.F": wrf / "phys/module_mp_thompson.F"}
    for name, source in sources.items():
        shutil.copyfile(source, out / name)
    shutil.copyfile(old / "thompgh_stub_modules.f90", out / "stubs.f90")
    driver = (old / "thompgh_oracle_driver.f90").read_text()
    begin = driver.index("  ! ---- initialize Thompson module constants")
    end = driver.index("  ! save inputs", begin)
    driver = driver[:begin] + """  ! Production physics_init.F:4565: BOTH aerosol and hail switches.
  CALL thompson_init(HGT=hgt3d, ORHO=1.0/p, &
                     NWFA2D=nwfa2d,NWFA=nwfa,NIFA=nifa,NBCA=nbca, &
                     NG=ng,wif_input_opt=1,is_start=.TRUE., &
                     FRC_URB2D=nwfa2d,DX=3000.,DY=3000., &
                     ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
                     ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
                     its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte)
\n""" + driver[end:]
    # The init's its:ite-1 slice needs one active column. The driver still
    # computes MIN(ite,ide-1)=1: no additional column is integrated.
    driver = driver.replace("its=1, ite=1, jts=1, jte=1", "its=1, ite=2, jts=1, jte=2")
    # physics_init.F:988-1024 does not include THOMPSONGH in the optional
    # effective-radius admission predicate, even when use_mp_re=1.
    driver = driver.replace("has_reqc=1; has_reqi=1; has_reqs=1", "has_reqc=0; has_reqi=0; has_reqs=0")
    # ORHO's production input is inverse density, not pressure. With supplied
    # nonzero aerosols no generated profile branch reads it; nevertheless bind
    # the exact caller value from Thompson's thermodynamic density.
    driver = driver.replace("ORHO=1.0/p", "ORHO=(287.04*(th*pii)*(qv+0.622))/(0.622*p)")
    driver = driver.replace("  CALL dump_3d('NG_OUT',", "  CALL dump_3d('NWFA_OUT', nwfa, .FALSE.)\n  CALL dump_3d('NIFA_OUT', nifa, .FALSE.)\n  CALL dump_3d('NBCA_OUT', nbca, .FALSE.)\n  CALL dump_3d('NG_OUT',")
    (out / "driver.f90").write_text(driver)
    # Reuse the actual WRF-produced snow/freezing inputs; mp38 collision
    # data are generated once by the unmodified original scheme.
    reference = wrf / "test/em_real/oracle_run_v090_mp28"
    for name in ("qr_acr_qsV2.dat", "freezeH2O.dat"):
        target = out / name
        if not target.exists():
            target.symlink_to(reference / name)
    (out / "CCN_ACTIVATE.BIN").symlink_to(wrf / "run/CCN_ACTIVATE.BIN")
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    common = ["-O2", "-cpp", "-ffree-line-length-none", "-ffixed-line-length-none", "-fallow-argument-mismatch",
              "-fconvert=big-endian", "-ffp-contract=off", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    objects = []
    for name in ("stubs.f90", "module_model_constants.F", "libmassv.F", "module_mp_radar.F", "module_mp_thompson.F", "driver.f90"):
        form = "-ffixed-form" if name == "libmassv.F" else "-ffree-form"
        subprocess.run([compiler, *common, form, "-c", name], check=True, cwd=out)
        objects.append(str(Path(name).with_suffix(".o")))
    subprocess.run([compiler, *common, *objects, "-o", "oracle"], check=True, cwd=out)
    receipt = {"scope": "True mp38 init flags: hail and aerosol, aer_init_opt0,wif_input_opt1, supplied nonzero aerosols, one column",
               "source_sha256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in (*sources, "driver.f90", "stubs.f90")},
               "source_files_verbatim": list(sources), "flags": common,
               "tables": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ("qr_acr_qsV2.dat", "freezeH2O.dat", "CCN_ACTIVATE.BIN")}}
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for case in range(1, 7):
        with (out / f"case{case}.txt").open("w") as stream:
            subprocess.run([str(out / "oracle"), str(case)], check=True, cwd=out, stdout=stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wrf", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.wrf, args.out)
