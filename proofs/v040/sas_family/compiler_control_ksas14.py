"""Independent original-source FMA control; primary receipt stays unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from evidence_gate import require_finite
from probe_ksas14 import parse,compare_fields

COMPILER="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
OUTPUTS=("HBOT","HTOP","RAINCV","PRATEC","RTHCUTEN","RQVCUTEN","RQCCUTEN","RQICUTEN","RUCUTEN","RVCUTEN")


def run(reference,work):
    work.mkdir(parents=True,exist_ok=True)
    build=work/"build";build.mkdir(exist_ok=True)
    original=reference/"build_r4"
    source=original/"module_cu_ksas.F"
    shutil.copyfile(source,build/source.name)
    flags=(original/"flags.txt").read_text().split()+["-march=native","-ffp-contract=fast"]
    subprocess.run([COMPILER,*flags,"-c",source.name],cwd=build,check=True,timeout=120)
    subprocess.run([COMPILER,"-O2","module_cu_ksas.o",str(original/"ksas14_oracle.o"),"-o","oracle"],
                   cwd=build,check=True,timeout=120)
    controls=[];primary=[]
    for case in range(1,6):
        path=work/f"case{case}_r4.txt"
        path.write_bytes(subprocess.check_output([str(build/"oracle"),str(case)],timeout=30))
        controls.append(parse(path))
        primary.append(parse(reference/f"case{case}_r4.txt"))
    inp={k:np.stack([d[k] for d in controls]) for k in controls[0]}
    old={k:np.stack([d[k] for d in primary]) for k in primary[0]}
    require_finite(**inp)
    require_finite(**old)
    assert all(np.array_equal(inp[k],old[k]) for k in inp if k not in OUTPUTS), "input mismatch"
    with np.load(reference/"candidate_r4.npz") as saved:
        actual=dict(saved)
    fields=compare_fields(actual,inp,"r4")
    # Disclose independent original-Fortran compiler variation too.
    spread={k:float(np.max(np.abs(inp[k]-old[k]))) for k in OUTPUTS}
    report={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"flags":flags,
            "original_inputs_exact":True,"unchanged_bounds":True,"fields":fields,
            "pass":all(v["pass"] for v in fields.values()),"fortran_FMA_vs_primary_maxabs":spread,
            "candidate_sha256":hashlib.sha256((reference/"candidate_r4.npz").read_bytes()).hexdigest(),
            "scope":"compiler-control diagnostic; original primary gate retained; no product ISA/precision change"}
    (work/"control.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("work",type=Path)
    a=p.parse_args();run(a.reference.resolve(),a.work.resolve())
