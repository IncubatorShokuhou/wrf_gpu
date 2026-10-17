"""WRITE-only staged NSAS core outputs; original full CU_NSAS stdout preserved."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

COMPILER="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
FIELDS={"T":"t1","QV":"q1","QI":"qi2","QC":"qc2","U":"u1","V":"v1","RAIN":"rain",
        "DEL":"del","PRSL":"prsll","Z":"zll","DOT":"dot"}


def stage(reference,work,modes=("r4","r8"),phases=False):
    work.mkdir(parents=True,exist_ok=True)
    report={"original_source":"module_cu_nsas.F","write_only":True,"modes":{}}
    phase_report={}
    for mode in modes:
        old=reference/f"build_{mode}"
        build=work/mode;build.mkdir(exist_ok=True)
        source=(old/"module_cu_nsas.F").read_text()
        anchor="     call nscv2d("
        pos=source.index(anchor)
        writes="".join(f"     write(*,'(A,*(ES27.17E3))') 'PROBE {key} ',{var}\n"
                       for key,var in FIELDS.items())
        writes+="     write(*,'(A,*(I8))') 'PROBE KBOT ',kbot\n"
        writes+="     write(*,'(A,*(I8))') 'PROBE KTOP ',ktop\n"
        writes+="     write(*,'(A,*(I8))') 'PROBE ICPS ',icps\n"
        changed=source[:pos]+writes+source[pos:]
        anchor2="   kbm = min(kbm,kte)"
        pos=changed.index(anchor2)+len(anchor2)
        changed=changed[:pos]+"\n   write(*,'(A,3I8)') 'PROBE LIMITS ',kbm,kbmax,kmax"+changed[pos:]
        if phases:
            def write_phase(key,expr):
                return f"   write(*,'(A,*(ES27.17E3))') 'PHASE {key} ',{expr}\n"
            pos=changed.index("! look for convective cloud base")
            pos=changed.rfind("\n",0,pos)+1
            extra="".join(write_phase(key,f"{var}(:,kts:kmax)") for key,var in
                          (("ENV_TO","to"),("ENV_QO","qo"),("ENV_QESO","qeso"),("ENV_HEO","heo"),("ENV_HESO","heso")))
            changed=changed[:pos]+extra+changed[pos:]
            import re
            deep_begin=changed.index("   subroutine nsas2d(")
            deep_end=changed.index("   end subroutine nsas2d",deep_begin)
            matches=[m for m in re.finditer(r"^   do i = its,ite\s*\n     if\(cnvflg\(i\)\.and\.aa1\(i\)\.le\.0\.\)",changed,re.M)
                     if deep_begin<m.start()<deep_end]
            assert len(matches)==2
            for match,tag in reversed(list(zip(matches,("UP","DD")))):
                extra=write_phase("AA1_"+tag,"aa1")
                variables=(("ETA","eta(:,kts:kmax)"),("XLAMB","xlamb(:,kts:kmax-1)"),
                           ("HCKO","hcko(:,kts:kmax)"),("QCKO","qcko(:,kts:kmax)"),("PWO","pwo(:,kts:kmax)")) if tag=="UP" else (("EDTO","edto"),("PWEVO","pwevo"))
                extra+="".join(write_phase(key,expr) for key,expr in variables)
                changed=changed[:match.start()]+extra+changed[match.start():]
            pos=changed.index("! restore t0 and qo to t1 and q1")
            pos=changed.rfind("\n",0,pos)+1
            extra="".join(write_phase(key,var) for key,var in
                          (("XAA0","xaa0"),("ACRT","acrt"),("ACRTFCT","acrtfct"),("DTCONV","dtconv"),("F","f"),("XK","xk"),("XMB","xmb")))
            changed=changed[:pos]+extra+changed[pos:]
        assert "".join(line for line in changed.splitlines(keepends=True) if "'PROBE " not in line and "'PHASE " not in line)==source
        (build/"module_cu_nsas.F").write_text(changed)
        flags=(old/"flags.txt").read_text().split()
        (work/"tmp").mkdir(exist_ok=True)
        env=dict(os.environ,TMPDIR=str(work/"tmp"))
        subprocess.run([COMPILER,*flags,"-c","module_cu_nsas.F"],cwd=build,check=True,timeout=120,env=env)
        subprocess.run([COMPILER,"-O2","module_cu_nsas.o",str(old/"nsas96_oracle.o"),"-o","staged"],
                       cwd=build,check=True,timeout=120,env=env)
        columns=[]
        for case in range(1,6):
            raw=subprocess.check_output([str(build/"staged"),str(case)],timeout=30)
            numeric=b"";data={};observed={}
            for line in raw.splitlines(keepends=True):
                if line.startswith(b"PROBE "):
                    _,key,*values=line.decode().split()
                    data[key]=np.asarray([float(v) for v in values])
                elif line.startswith(b"PHASE "):
                    _,key,*values=line.decode().split()
                    observed[key]=[float(v) for v in values]
                else:numeric+=line
            assert numeric==(reference/f"case{case}_{mode}.txt").read_bytes(), "full stdout changed"
            assert set(data)==set(FIELDS)|{"KBOT","KTOP","ICPS","LIMITS"}
            columns.append(data)
            phase_report[f"case{case}_{mode}"]=observed
            (work/f"case{case}_{mode}.txt").write_bytes(raw)
        arrays={k:np.stack([d[k] for d in columns]) for k in columns[0]}
        np.savez_compressed(work/f"deep_{mode}.npz",**arrays)
        report["modes"][mode]={"original_stdout_exact":True,"cases":5,
            "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
            "write_only_sha256":hashlib.sha256(changed.encode()).hexdigest(),
            "active":int(np.count_nonzero(arrays["ICPS"]))}
    (work/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    if phases:(work/"phases.json").write_text(json.dumps(phase_report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("work",type=Path)
    p.add_argument("--mode",choices=("r4","r8"));p.add_argument("--phases",action="store_true")
    a=p.parse_args();stage(a.reference.resolve(),a.work.resolve(),(a.mode,) if a.mode else ("r4","r8"),a.phases)
