"""Compare original local FPVS with the native JAX helper at observed temperatures."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from probe_ksas14 import parse


def run(reference,phase_file,work):
    work.mkdir(parents=True,exist_ok=True);(work/"tmp").mkdir(exist_ok=True)
    phases=json.loads(phase_file.read_text())
    temps=[np.asarray(phases[f"case{i}_r4"]["ENV_TO"],np.float32) for i in range(1,6)]
    temps+=[parse(reference/f"case{i}_r4.txt")["T"].astype(np.float32) for i in range(1,6)]
    temps=np.concatenate(temps)
    here=Path(__file__).parent
    driver=(here/"nsas96_oracle.F90").read_text()
    params=driver[driver.index("  REAL, PARAMETER :: G"):driver.index("  INTEGER, PARAMETER :: KX")]
    program="PROGRAM saturation\nUSE module_cu_nsas, ONLY: fpvs\nIMPLICIT NONE\n"+params+"""
INTEGER :: n,i
REAL :: temp
READ(*,*) n
DO i=1,n
 READ(*,*) temp
 WRITE(*,'(ES27.17E3)') fpvs(temp,1,R_D,R_V,CPV,CLIQ,CICE,XLV,XLS,PSAT,273.15)
END DO
END PROGRAM saturation
"""
    (work/"query.f90").write_text(program)
    original=reference/"build_r4"
    compiler="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    env=dict(os.environ,TMPDIR=str(work/"tmp"))
    subprocess.run([compiler,"-O2","-ffree-form","-ffree-line-length-none","-I",str(original),
                    str(work/"query.f90"),str(original/"module_cu_nsas.o"),"-o",str(work/"query")],
                   check=True,timeout=60,cwd=work,env=env)
    data=str(len(temps))+"\n"+"\n".join(repr(float(t)) for t in temps)+"\n"
    raw=subprocess.check_output([str(work/"query")],input=data.encode(),timeout=30)
    expected=np.asarray([float(line) for line in raw.splitlines()],np.float32)
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.cumulus_ksas14 import _fpvs
    jax.config.update("jax_enable_x64",True)
    assert jax.devices()[0].platform=="cpu"
    start=time.monotonic()
    actual=np.asarray(jax.jit(lambda t:_fpvs(t,"r4"))(jnp.asarray(temps)))
    delta=np.abs(actual.astype(np.float64)-expected)
    ulp=np.abs(np.spacing(np.maximum(np.abs(actual),np.abs(expected))))
    report={"backend":"cpu","points":len(temps),"exact":int(np.count_nonzero(actual==expected)),
            "finite":bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
            "max_abs":float(delta.max()),"max_ulp":float(np.max(delta/ulp)),
            "elapsed_compile_execute_s":time.monotonic()-start,
            "first_differences":np.flatnonzero(actual!=expected)[:20].tolist(),
            "original_source_sha256":hashlib.sha256((original/"module_cu_nsas.F").read_bytes()).hexdigest(),
            "scope":"standalone primitive; same original observed ENV_TO and input T; no product change"}
    np.savez_compressed(work/"comparison.npz",temperature=temps,expected=expected,actual=actual)
    (work/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("reference",type=Path);p.add_argument("phases",type=Path)
    p.add_argument("work",type=Path);a=p.parse_args();run(a.reference.resolve(),a.phases.resolve(),a.work.resolve())
