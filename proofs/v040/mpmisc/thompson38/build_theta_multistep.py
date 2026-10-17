"""Restore lossless original tables and capture public theta for three steps.

No quadrature or pristine-body edits. Every restored raw byte is certified
against the original table SHA before the original Fortran module is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def main(repo, wrf, output):
    start=time.monotonic();base=output/"base";base.mkdir(parents=True,exist_ok=True)
    proof=repo/"proofs/v040/mpmisc/thompson38"
    names=("tcg_racg","tmr_racg","tcr_gacr","tnr_racg","tnr_gacr")
    shape=(37,37,9,37,37);length=int(np.prod(shape))*8
    raw=base/"qr_acr_qg_mp38V1.dat"
    with raw.open("wb") as stream:
        for name in names:
            values=np.empty(shape,dtype=">f8",order="F")
            for density in range(9):
                with np.load(repo/f"src/gpuwrf/physics/thompson_hail/collision_tables_{density}.npz",allow_pickle=False) as packed:
                    values[:,:,density,:,:]=packed[name]
            stream.write(struct.pack(">i",length))
            stream.write(values.ravel(order="F").view(np.uint8))
            stream.write(struct.pack(">i",length))
            del values
    expected=json.loads((proof/"collision_tables.json").read_text())["source_sha256"]
    assert sha(raw)==expected
    source_map={"module_model_constants.F":wrf/"share/module_model_constants.F",
                "libmassv.F":wrf/"frame/libmassv.F","module_mp_radar.F":wrf/"phys/module_mp_radar.F",
                "module_mp_thompson.F":wrf/"phys/module_mp_thompson.F",
                "stubs.f90":repo/"proofs/v018/mp_oracles/thompgh/oracle/thompgh_stub_modules.f90"}
    manifest=json.loads((proof/"original_build.json").read_text())["source_sha256"]
    for name,source in source_map.items():
        assert sha(source)==manifest[name],name
        shutil.copyfile(source,base/name)
    reference=wrf/"test/em_real/oracle_run_v090_mp28"
    for name in ("qr_acr_qsV2.dat","freezeH2O.dat"):(base/name).symlink_to(reference/name)
    (base/"CCN_ACTIVATE.BIN").symlink_to(wrf/"run/CCN_ACTIVATE.BIN")
    compiler="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O2","-cpp","-ffree-line-length-none","-ffixed-line-length-none","-fallow-argument-mismatch","-fconvert=big-endian","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    for name in ("stubs.f90","module_model_constants.F","libmassv.F","module_mp_radar.F","module_mp_thompson.F"):
        form="-ffixed-form" if name=="libmassv.F" else "-ffree-form"
        subprocess.run([compiler,*flags,form,"-c",name],cwd=base,check=True)
    # The existing project-authored three-step caller is retained verbatim,
    # apart from adding one public output field to its dump.
    fresh=(proof/"fresh_driver.f90").read_text()
    needle="  CALL dump_3d('T_OUT',    th, .TRUE.)"
    assert fresh.count(needle)==1
    fresh=fresh.replace(needle,needle+"\n  CALL dump_3d('TH_OUT', th, .FALSE.)")
    (output/"fresh_driver.f90").write_text(fresh)
    helper=output/"build_multistep_oracle.py"
    helper.write_text((proof/"build_multistep_oracle.py").read_text())
    subprocess.run(["python",str(helper),str(base),str(output/"caller")],cwd=output,check=True)
    result=dict(scope="Original three-own-step caller plus directTH_OUT only, lossless table restoration/no quadrature",
        raw_table_sha256=sha(raw),raw_table_original_SHA_BYTE=True,original_sources_SHA_BYTE=True,
        flags=flags,wall_s=time.monotonic()-start)
    (output/"restore_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("repo",type=Path);parser.add_argument("wrf",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();main(args.repo,args.wrf,args.output)
