"""Pristine OSAS oracle with actual SYSTEM_CLOCK random draws recorded."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

PHYS=Path("<USER_HOME>/src/wrf_pristine/WRF/phys")
COMPILER="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"


def run(work,inputs):
    work.mkdir(parents=True,exist_ok=True)
    (work/"tmp").mkdir(exist_ok=True)
    env=dict(os.environ,TMPDIR=str(work/"tmp"))
    report={"source":"original WRF4.7.1 CU_OSAS","random_policy":"actual original SYSTEM_CLOCK/RANDOM_NUMBER draws recorded after generation; NO REGTEST","write_only":True,"arms":{}}
    for mode in ("r4","r8"):
        build=work/f"build_{mode}";build.mkdir(exist_ok=True)
        modules=("module_gfs_machine","module_gfs_physcons","module_gfs_funcphys","module_cu_osas")
        for name in modules:shutil.copyfile(PHYS/f"{name}.F",build/f"{name}.F")
        path=build/"module_cu_osas.F"
        source=path.read_text()
        pattern=r"^     call random_number\(XKT2\)$"
        assert len(re.findall(pattern,source,re.M))==1
        observed=re.sub(pattern,lambda m:m.group(0)+"\n     write(*,'(A,*(ES27.17E3))') 'DRAW_XKT2 ',XKT2",source,flags=re.M)
        assert "\n".join(line for line in observed.split("\n") if "'DRAW_XKT2 " not in line)==source
        path.write_text(observed)
        shutil.copyfile(Path(__file__).with_name("osas95_oracle.F90"),build/"osas95_oracle.F90")
        flags=["-O2","-ftree-vectorize","-funroll-loops","-w","-ffree-form","-ffree-line-length-none","-cpp","-DEM_CORE=1","-DNMM_CORE=0"]
        if mode=="r8":flags+=["-fdefault-real-8","-fdefault-double-8"]
        for name in (*modules,"osas95_oracle"):
            suffix="F90" if name=="osas95_oracle" else "F"
            subprocess.run([COMPILER,*flags,"-c",f"{name}.{suffix}"],cwd=build,check=True,timeout=120,env=env)
        subprocess.run([COMPILER,"-O2",*[f"{n}.o" for n in (*modules,"osas95_oracle")],"-o","oracle"],cwd=build,check=True,timeout=120,env=env)
        report[mode]={"flags":flags,"original_source_sha256":hashlib.sha256(source.encode()).hexdigest(),"write_only_sha256":hashlib.sha256(observed.encode()).hexdigest()}
        for inp in inputs:
            raw=subprocess.check_output([str(build/"oracle")],input=inp.read_bytes(),cwd=build,timeout=120,env=env)
            draws=[line.split()[1:] for line in raw.splitlines() if line.startswith(b"DRAW_XKT2 ")]
            assert len(draws)==1
            output=work/f"{inp.stem}_{mode}.out";output.write_bytes(raw)
            report["arms"][f"{inp.stem}_{mode}"]={"draws":len(draws[0]),"input_sha256":hashlib.sha256(inp.read_bytes()).hexdigest(),"output_sha256":hashlib.sha256(raw).hexdigest()}
    (work/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("work",type=Path);p.add_argument("inputs",nargs="+",type=Path)
    a=p.parse_args();run(a.work.resolve(),[p.resolve() for p in a.inputs])
