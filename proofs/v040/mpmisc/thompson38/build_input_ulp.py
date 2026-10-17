"""Independent original-WRF +/-NEAREST response ensemble, no candidate data.

Each field is perturbed independently over the column, with zero entries
held zero, following the lane's pre-existing Ferrier qualification recipe.
Tables initialize once per original caller process, not once per variant.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np


FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "ng", "qb", "nwfa", "nifa", "th", "p", "pii", "w", "dz")
CODES = dict(qv="QV", qc="QC", qr="QR", qi="QI", qs="QS", qg="QG", ni="NI", nr="NR", nc="NC", ng="NG", qb="QB", nwfa="NWFA", nifa="NIFA", th="T")


def parser():
    # Extract only the pure parser; importing the compiled checker would
    # unnecessarily start a JAX process during this Fortran-only arm.
    tree=ast.parse(Path(__file__).with_name("check_columns.py").read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="parse")
    env=dict(np=np,re=re)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"reference_parser","exec"),env)
    return env["parse"]


def main(base, fresh, threshold, output):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    read=parser();refs={case:read((base if case<=6 else fresh if case<=9 else threshold)/f"case{case}.txt") for case in range(1,12)}
    for case,ref in refs.items():
        columns=ref["columns"]
        temp=np.asarray(columns["T_IN"],np.float32);pii=np.asarray(columns["PII"],np.float32)
        theta=temp/pii
        for k in range(40):
            if theta[k]*pii[k]!=temp[k]:
                options=(np.nextafter(theta[k],np.float32(-np.inf)),np.nextafter(theta[k],np.float32(np.inf)))
                theta[k]=next(x for x in options if x*pii[k]==temp[k])
        values={field:np.asarray(columns[code+"_IN"],np.float32) for field,code in CODES.items()}
        values.update(th=theta,p=np.asarray(columns["P"],np.float32),pii=pii,w=np.asarray(columns.get("W_IN",np.zeros(40)),np.float32),dz=np.asarray(columns["DELZ"],np.float32),nbca=np.asarray(columns["NBCA_IN"],np.float32))
        lines=[str(ref["scalars"]["DT"])]+[" ".join(format(float(values[field][k]),".17e") for field in (*FIELDS,"nbca")) for k in range(40)]
        (output/f"input{case}.txt").write_text("\n".join(lines)+"\n")
    source=Path(__file__).with_name("fresh_driver.f90").read_text()
    source=source.replace("INTEGER :: k, case_id,", "INTEGER :: variant, ivar, k, case_id,")
    source=source.replace("  CHARACTER(LEN=32) :: arg", "  CHARACTER(LEN=32) :: arg\n  CHARACTER(LEN=64) :: input_path\n  REAL :: direction\n  REAL :: "+",".join("x_"+field for field in (*FIELDS,"nbca")))
    restore="  WRITE(input_path,'(A,I0,A)') 'input',case_id,'.txt'\n  OPEN(97,FILE=TRIM(input_path),STATUS='OLD')\n  READ(97,*) DT\n  DO k=kts,kte\n    READ(97,*) "+",".join("x_"+field for field in (*FIELDS,"nbca"))+"\n"
    restore+="\n".join(f"    {field}(:,k,:)=x_{field}" for field in (*FIELDS,"nbca"))+"\n  END DO\n  CLOSE(97)\n"
    source=source.replace("  CALL build_column(case_id)",restore)
    block="  DO variant=0,36\n"+restore+"  IF (variant>0) THEN\n    ivar=(variant+1)/2\n    direction=1.\n    IF(MOD(variant,2)==1) direction=-1.\n    SELECT CASE(ivar)\n"
    for index,field in enumerate(FIELDS,1):
        block+=f"    CASE({index})\n      WHERE({field}/=0.) {field}=NEAREST({field},direction)\n"
    block+="    END SELECT\n  END IF\n  WRITE(*,'(A,I0)') 'VARIANT=',variant\n"
    source=source.replace("  ! save inputs (the EXACT pre-call values the scheme receives).",block+"  ! save inputs (the EXACT pre-call values the scheme receives).")
    source=source.replace("\nCONTAINS", "\n  END DO\n\nCONTAINS")
    caller=output/"ulp_driver.f90";caller.write_text(source)
    compiler="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O2","-cpp","-ffree-form","-ffree-line-length-none","-fallow-argument-mismatch","-fconvert=big-endian","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([compiler,*flags,"-I"+str(base),"-c",str(caller),"-o",str(output/"caller.o")],cwd=output,check=True)
    dependencies=[str(base/(name+".o")) for name in ("stubs","module_model_constants","libmassv","module_mp_radar","module_mp_thompson")]
    subprocess.run([compiler,*flags,*dependencies,str(output/"caller.o"),"-o",str(output/"oracle")],cwd=output,check=True)
    for name in ("qr_acr_qg_mp38V1.dat","qr_acr_qsV2.dat","freezeH2O.dat","CCN_ACTIVATE.BIN"):
        (output/name).symlink_to(base/name)
    intervals={};baseline_checks=[]
    for case,reference in refs.items():
        raw=output/f"case{case}.raw"
        with raw.open("w") as stream:subprocess.run([str(output/"oracle"),str(case)],cwd=output,stdout=stream,check=True)
        text=raw.read_text();variants=[]
        for variant in range(37):
            content=text.split(f"VARIANT={variant}\n",1)[1].split("VARIANT=",1)[0]
            parsed=output/"parse.txt";parsed.write_text(content);variants.append(read(parsed))
        checks={name:np.array_equal(np.asarray(value,np.float32),np.asarray(variants[0]["columns"][name],np.float32)) for name,value in reference["columns"].items() if name.endswith("_IN") or name in ("P","PII","DELZ")}
        assert all(checks.values()),(case,checks)
        output_checks={name:np.array_equal(np.asarray(value,np.float32),np.asarray(variants[0]["columns"][name],np.float32)) for name,value in reference["columns"].items() if name.endswith("_OUT")}
        assert all(output_checks.values()),(case,output_checks)
        baseline_checks.append(dict(case=case,input_fields_BYTE=checks,output_fields_BYTE=output_checks))
        for name,value in {**reference["columns"],**reference["scalars"]}.items():
            if not(name.endswith("_OUT") or name in ("RAINNC","RAINNCV","SNOWNC","SNOWNCV","GRAUPELNC","GRAUPELNCV","SR")):continue
            vectors=np.asarray([(v["columns"] if name in v["columns"] else v["scalars"])[name] for v in variants],np.float32)
            assert np.isfinite(vectors).all(),(case,name)
            intervals[f"case{case}__{name}__lower"]=vectors.min(axis=0)
            intervals[f"case{case}__{name}__upper"]=vectors.max(axis=0)
    np.savez_compressed(output/"original_input_ulp_intervals.npz",**intervals)
    result=dict(scope="Independent original-WRF +/-1 input-ULP per-field response intervals, no candidate-derived bounds",variants=37,cases=11,fields=FIELDS,zero_entries_held=True,
        caller_sha256=hashlib.sha256(caller.read_bytes()).hexdigest(),original_source_sha256=hashlib.sha256((base/"module_mp_thompson.F").read_bytes()).hexdigest(),
        interval_sha256=hashlib.sha256((output/"original_input_ulp_intervals.npz").read_bytes()).hexdigest(),baseline=baseline_checks,wall_s=time.monotonic()-started,flags=flags)
    (output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("base",type=Path);p.add_argument("fresh",type=Path);p.add_argument("threshold",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();main(args.base,args.fresh,args.threshold,args.output)
