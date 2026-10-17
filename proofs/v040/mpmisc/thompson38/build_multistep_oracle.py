"""Three original own-steps, held emissions, theta and accumulated surfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main(base, output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = Path(__file__).with_name("fresh_driver.f90").read_text()
    source = source.replace("INTEGER :: k, case_id,", "INTEGER :: step, k, case_id,")
    source = source.replace("nwfa2d = 0.0; nifa2d = 0.0; nbca2d = 0.0", "nwfa2d = 123.25; nifa2d = 0.75; nbca2d = 0.0")
    reset = "  RAINNC=0.; RAINNCV=0.; SR=0.\n  SNOWNC=0.; SNOWNCV=0.; GRAUPELNC=0.; GRAUPELNCV=0."
    assert source.count(reset) == 1
    source = source.replace(reset, "")
    marker = "  ! save inputs (the EXACT pre-call values the scheme receives)."
    source = source.replace(marker, """  ! Nonzero carried accumulators exercise the wrapper's original REAL
  ! left-association, including previous accumulation FIRST (:1299-1304).
  RAINNC=7.25; RAINNCV=0.; SR=0.
  SNOWNC=3.25; SNOWNCV=0.; GRAUPELNC=2.25; GRAUPELNCV=0.
  DO step=1,3
  WRITE(*,'(A,I0)') 'STEP=', step
  WRITE(*,'(A,ES23.15)') 'RAINNC_IN=', RAINNC(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNC_IN=', SNOWNC(1,1)
  WRITE(*,'(A,ES23.15)') 'GRAUPELNC_IN=', GRAUPELNC(1,1)
  WRITE(*,'(A,ES23.15)') 'NWFA2D=', nwfa2d(1,1)
  WRITE(*,'(A,ES23.15)') 'NIFA2D=', nifa2d(1,1)
""" + marker)
    source = source.replace("itimestep=1,", "itimestep=step,")
    source = source.replace("\nCONTAINS", "\n  END DO\n\nCONTAINS")
    caller = output / "multistep_driver.f90"
    caller.write_text(source)
    for name in ("qr_acr_qg_mp38V1.dat", "qr_acr_qsV2.dat", "freezeH2O.dat", "CCN_ACTIVATE.BIN"):
        (output/name).symlink_to(base/name)
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags = ["-O2", "-cpp", "-ffree-form", "-ffree-line-length-none", "-fallow-argument-mismatch",
             "-fconvert=big-endian", "-ffp-contract=off", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
    subprocess.run([compiler, *flags, "-I"+str(base), "-c", str(caller), "-o", str(output/"caller.o")], cwd=output, check=True)
    deps = [str(base/(name+".o")) for name in ("stubs", "module_model_constants", "libmassv", "module_mp_radar", "module_mp_thompson")]
    subprocess.run([compiler, *flags, *deps, str(output/"caller.o"), "-o", str(output/"oracle")], cwd=output, check=True)
    with (output/"three_steps.txt").open("w") as stream:
        subprocess.run([str(output/"oracle"), "9"], cwd=output, stdout=stream, check=True)
    original = output/"three_steps.txt"
    for step in range(1,4):
        text = original.read_text().split("STEP="+str(step)+"\n",1)[1].split("STEP=",1)[0]
        (output/f"step{step}.txt").write_text(text)
    receipt = dict(scope="Unmodified original WRF three own-steps with held positive emissions and nonzero accumulated surfaces",
                   case=9, steps=3, dt=18, flags=flags, caller_sha256=hashlib.sha256(caller.read_bytes()).hexdigest(),
                   original_source_sha256=hashlib.sha256((base/"module_mp_thompson.F").read_bytes()).hexdigest(),
                   module_object_sha256=hashlib.sha256((base/"module_mp_thompson.o").read_bytes()).hexdigest(),
                   wall_s=time.monotonic()-started)
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    main(args.base,args.output)
