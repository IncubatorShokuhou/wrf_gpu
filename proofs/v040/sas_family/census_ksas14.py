"""WRITE-only pristine KSAS trigger census, verified against original stdout."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

STAGES=("kbcon","cinpcri","dthk","cina","cthk","updraft_work","downdraft_work","closure")
COMPILER="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"


def run(reference,work):
    work.mkdir(parents=True,exist_ok=True)
    report={"source":"WRF4.7.1 module_cu_ksas.F","write_only":True,"arms":{}}
    for mode in ("r4","r8"):
        original=reference/f"build_{mode}"
        build=work/mode
        build.mkdir(exist_ok=True)
        source=(original/"module_cu_ksas.F").read_text()
        begin=source.index("   subroutine nsas2d(")
        end=source.index("   end subroutine nsas2d",begin)
        body=source[begin:end]
        counter=0
        def inject(match):
            nonlocal counter
            stage=STAGES[counter]
            counter+=1
            return f"   write(*,'(A,I8)') 'CENSUS {stage} ',count(cnvflg)\n"+match.group(0)
        changed_body=re.sub(r"^   if\s*\(totflg\) return.*$",inject,body,flags=re.M)
        assert counter==len(STAGES)
        assert re.sub(r"^   write\(.*\n","",changed_body,flags=re.M)==body
        changed=source[:begin]+changed_body+source[end:]
        (build/"module_cu_ksas.F").write_text(changed)
        flags=(original/"flags.txt").read_text().split()
        subprocess.run([COMPILER,*flags,"-c","module_cu_ksas.F"],cwd=build,check=True,timeout=120)
        subprocess.run([COMPILER,"-O2","module_cu_ksas.o",str(original/"ksas14_oracle.o"),"-o","census"],
                       cwd=build,check=True,timeout=120)
        report[mode+"_source_sha256"]=hashlib.sha256(source.encode()).hexdigest()
        report[mode+"_instrumented_sha256"]=hashlib.sha256(changed.encode()).hexdigest()
        for case in range(1,6):
            raw=subprocess.check_output([str(build/"census"),str(case)],timeout=30)
            stages={}
            numeric=b""
            for line in raw.splitlines(keepends=True):
                if line.startswith(b"CENSUS "):
                    _,stage,n=line.decode().split()
                    stages[stage]=int(n)
                else:
                    numeric+=line
            assert numeric==(reference/f"case{case}_{mode}.txt").read_bytes()
            report["arms"][f"case{case}_{mode}"]={"populations":stages,"original_stdout_byte_exact":True}
    (work/"census.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("work",type=Path)
    a=p.parse_args();run(a.reference.resolve(),a.work.resolve())
