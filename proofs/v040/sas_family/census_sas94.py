"""Source-only WRITE instrumentation of copied pristine SASCNVN; no JAX run.

The seven filter-stage populations are collected from the original Fortran
module, and every original output byte must match the uninstrumented oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

STAGES = ("kbcon", "cincr", "dthk", "cthk", "updraft_work", "downdraft_work", "closure")


def run(reference, work, modes=("r4","r8"), masks=False, build_reference=None):
    work.mkdir(parents=True,exist_ok=True)
    report = {"source": "WRF 4.7.1 module_cu_sas.F SASCNVN", "write_only": True, "arms": {}}
    compiler = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    for mode in modes:
        old = (build_reference or reference)/f"build_{mode}"
        build = work/mode
        build.mkdir(exist_ok=True)
        source = (old/"module_cu_sas.F").read_text()
        start = source.index("      subroutine sascnvn(")
        end = source.index("      end subroutine sascnvn",start)
        body = source[start:end]
        index = 0
        def inject(match):
            nonlocal index
            tag = STAGES[index]
            index += 1
            extra = (f"      write(*,'(A,*(I1))') 'FLAGS {tag} ',merge(1,0,cnvflg)\n" if masks else "")
            return f"      write(*,'(A,I8)') 'CENSUS {tag} ',count(cnvflg)\n"+extra+match.group(0)
        observed = re.sub(r"^      if\(totflg\) return.*$",inject,body,flags=re.M)
        assert index == 7 and re.sub(r"^      write\(.*\n","",observed,flags=re.M) == body
        changed = source[:start]+observed+source[end:]
        (build/"module_cu_sas.F").write_text(changed)
        for name in ("module_gfs_machine.mod","module_gfs_physcons.mod","module_gfs_funcphys.mod"):
            shutil.copyfile(old/name,build/name)
        flags = (old/"flags.txt").read_text().split()
        subprocess.run([compiler,*flags,"-c","module_cu_sas.F"],cwd=build,check=True,timeout=120)
        objects = [str(old/f"{name}.o") for name in ("module_gfs_machine","module_gfs_physcons","module_gfs_funcphys","sas94_oracle")]
        subprocess.run([compiler,"-O2","module_cu_sas.o",*objects,"-o","census"],cwd=build,check=True,timeout=120)
        report[mode+"_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        report[mode+"_instrumented_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
        for inp in sorted(reference.glob("*_shallow1.txt")):
            data = subprocess.check_output([str(build/"census")],input=inp.read_bytes(),cwd=build,timeout=120)
            lines = data.splitlines(keepends=True)
            census = {}
            flags = {}
            numeric = b""
            for line in lines:
                if line.startswith(b"CENSUS "):
                    _,tag,n = line.decode().split()
                    census[tag] = int(n)
                elif line.startswith(b"FLAGS "):
                    _,tag,bits = line.decode().split()
                    flags[tag] = bits
                else:
                    numeric += line
            expected = (reference/f"{inp.stem}_{mode}.out").read_bytes()
            assert numeric == expected, "instrumentation changed original output bytes"
            assert tuple(census) == STAGES
            report["arms"][inp.stem+"_"+mode] = {"populations": census, "original_output_byte_exact": True,
                                                         "output_sha256": hashlib.sha256(numeric).hexdigest()}
            if masks:
                report["arms"][inp.stem+"_"+mode]["flags"] = flags
            (work/f"{inp.stem}_{mode}.txt").write_bytes(data)
    (work/"census.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference",type=Path)
    p.add_argument("work",type=Path)
    p.add_argument("--mode",choices=("r4","r8"))
    p.add_argument("--masks",action="store_true")
    p.add_argument("--build-reference",type=Path)
    a = p.parse_args()
    run(a.reference.resolve(),a.work.resolve(),(a.mode,) if a.mode else ("r4","r8"),a.masks,
        a.build_reference.resolve() if a.build_reference else None)
