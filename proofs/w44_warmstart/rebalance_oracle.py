"""Extract unchanged rebalance_cycl arithmetic into a domain-free WRF oracle.

Only grid% qualification, domain indices and module/Registry declarations are
adapted. The original executable body, including both moisture branches and
PSFC, is compiled unchanged. Caller supplies immutable source + CPU guards.
"""
import ctypes as ct
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from gpuwrf.io.cycling_init import rebalance_cycling
from gpuwrf.io.netcdf_lock import Dataset

MOIST = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
ARRAYS = ("t_2", "moist", "mu_2", "mub", "pb", "phb", "alb", "c1f", "c2f", "c1h", "c2h",
          "c3f", "c4f", "c3h", "c4h", "dnw", "rdnw", "rdn")
OUTPUTS = ("P", "AL", "ALT", "P_HYD", "PH", "PH0", "PSFC")
# Frozen before the first oracle arm. P/PH match the established 0.2 absolute
# allowance; inverse density uses 2e-6 relative +2e-7 absolute (REAL libm).
TOLERANCES = {name: {"atol": 2e-7 if name in ("AL", "ALT") else .2,
                     "rtol": 2e-6 if name in ("AL", "ALT") else 0.} for name in OUTPUTS if name != "PH0"}


def build(pristine, artifacts):
    path = pristine / "WRF/dyn_em/start_em.F"
    source = path.read_text()
    start = source.index("   SUBROUTINE rebalance_cycl (")
    start = source.index("      DO j=jts,jte", start)
    end = source.index("      ips = its", start)
    body = source[start:end].replace("grid%", "")
    header = """subroutine wrf_rebalance(nx,ny,nf,ns,t_2,moist,mu_2,mub,pb,phb,alb, &
 c1f,c2f,c1h,c2h,c3f,c4f,c3h,c4h,dnw,rdnw,rdn,p_top,use_theta_m,hypsometric_opt, &
 p,al,alt,p_hyd,ph_2,ph0,psfc)
 implicit none
 integer,intent(in):: nx,ny,nf,ns,use_theta_m,hypsometric_opt
 integer,parameter:: PARAM_FIRST_SCALAR=2,P_QV=2
 integer:: num_moist,n_moist,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme
 integer:: its,ite,jts,jte,kts,kte,i,j,k,kk,ispe,ktf
 real,parameter:: r_d=287.,p1000mb=100000.,t0=300.,rvovrd=461.6/287.,cvpm=-2.5/3.5,g=9.81
 real,intent(in):: t_2(nx,nf,ny),moist(nx,nf,ny,ns+1),mu_2(nx,ny),mub(nx,ny)
 real,intent(in):: pb(nx,nf,ny),phb(nx,nf,ny),alb(nx,nf,ny)
 real,intent(in):: c1f(nf),c2f(nf),c1h(nf),c2h(nf),c3f(nf),c4f(nf),c3h(nf),c4h(nf)
 real,intent(in):: dnw(nf),rdnw(nf),rdn(nf),p_top
 real,intent(out):: p(nx,nf,ny),al(nx,nf,ny),alt(nx,nf,ny),p_hyd(nx,nf,ny)
 real,intent(out):: ph_2(nx,nf,ny),ph0(nx,nf,ny),psfc(nx,ny)
 real:: qtot,qvf,qvf1,qvf2,pfu,pfd,phm,z0,z1,z2,w1,w2
 ids=1;ide=nx+1;jds=1;jde=ny+1;kds=1;kde=nf
 its=1;ite=nx;jts=1;jte=ny;kts=1;kte=nf;num_moist=ns+1
 p=0.;al=0.;alt=0.;p_hyd=0.;ph_2=0.;ph0=0.;psfc=0.
"""
    generated = header + body + "\nend subroutine wrf_rebalance\n"
    adapted = artifacts / "rebalance.f90"
    adapted.write_text(generated)
    binary = artifacts / "rebalance.so"
    command = ["<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran", "-shared", "-fPIC", "-O0",
               "-ffp-contract=off", "-fcheck=all", "-ffree-line-length-none", str(adapted), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (artifacts / "compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return ct.CDLL(str(binary)), binary, {"original_path": str(path), "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "extracted_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "adapter_sha256": hashlib.sha256(generated.encode()).hexdigest(), "command": command}


def native_call(native, f, use_theta_m):
    nz,ny,nx = f["PB"].shape
    nf = nz+1
    def full(value):
        data=np.zeros((nx,nf,ny),np.float32,order="F")
        data[:,:value.shape[0],:]=value.transpose(2,0,1)
        return data
    def column(value):
        data=np.zeros(nf,np.float32);data[:len(value)]=value;return data
    args={"t_2":full(f["THM" if use_theta_m else "T"]),
          "moist":np.zeros((nx,nf,ny,len(MOIST)+1),np.float32,order="F"),
          "mu_2":np.asfortranarray(f["MU"].T),"mub":np.asfortranarray(f["MUB"].T)}
    for i,name in enumerate(MOIST,start=1): args["moist"][:,:nz,:,i]=f[name].transpose(2,0,1)
    for name in ("pb","phb","alb"): args[name]=full(f[name.upper()])
    for name in ARRAYS[7:]: args[name]=column(f[name.upper()])
    output=[np.zeros((nx,nf,ny),np.float32,order="F") for _ in range(6)]
    output.append(np.zeros((nx,ny),np.float32,order="F"))
    dims=[ct.c_int(n) for n in (nx,ny,nf,len(MOIST))]
    scalars=[ct.c_float(float(f["P_TOP"])),ct.c_int(use_theta_m),ct.c_int(2)]
    pointer=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_float))
    native.wrf_rebalance_(*map(ct.byref,dims),*[pointer(args[n]) for n in ARRAYS],*map(ct.byref,scalars),
                         *[pointer(a) for a in output])
    result={}
    for name,value in zip(OUTPUTS,output):
        result[name] = value.T.copy() if name=="PSFC" else value.transpose(1,2,0)[:(nf if name in ("PH","PH0") else nz)].copy()
    return result


def main():
    pristine,case,artifacts=map(Path,sys.argv[1:])
    artifacts.mkdir(exist_ok=False,parents=True)
    start=time.perf_counter()
    native,binary,metadata=build(pristine,artifacts)
    # Reuse the original init shim without importing the JAX model. The class
    # body is executed verbatim, with the same native C-array helper.
    shim_path=Path(__file__).resolve().parents[2]/"src/gpuwrf/io/wrf_init_math.py"
    shim_source=shim_path.read_text()
    node=next(n for n in ast.parse(shim_source).body if isinstance(n,ast.ClassDef) and n.name=="_WRFInitLibm32")
    namespace={"np":np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(shim_path),"exec"),namespace)
    libm=namespace["_WRFInitLibm32"]()
    metadata["libm_class_sha256"]=hashlib.sha256(ast.get_source_segment(shim_source,node).encode()).hexdigest()
    records=[]
    for initial in sorted(case.glob("wrfinput_d0[123]")):
        with Dataset(initial) as ds:
            names=("T","THM","MU","MUB","PB","PHB","ALB","C1F","C2F","C1H","C2H","C3F","C4F",
                   "C3H","C4H","DNW","RDNW","RDN","P_TOP",*MOIST)
            fields={name:np.asarray(ds[name][0],np.float32) for name in names}
        for use_theta_m in (0,1):
            before=time.perf_counter()
            candidate=rebalance_cycling(fields,moist_fields=MOIST,use_theta_m=use_theta_m,libm=libm)
            elapsed=time.perf_counter()-before
            reference=native_call(native,fields,use_theta_m)
            for name,value in candidate.items():
                error=np.abs(value.astype(float)-reference[name])
                limit=TOLERANCES[name]
                records.append({"file":str(initial),"use_theta_m":use_theta_m,"field":name,
                                "max_abs_error":float(error.max()),"byte_equal":value.tobytes()==reference[name].tobytes(),
                                "passed":bool(np.all(error<=limit["atol"]+limit["rtol"]*np.abs(reference[name]))),
                                "host_rebalance_seconds":elapsed})
    report={"passed":all(r["passed"] for r in records),"records":records,"source":metadata,"tolerances":TOLERANCES,
            "wall_seconds":time.perf_counter()-start,"scope":"original WRF rebalance_cycl on real 0227 cold arrays; cloudy-source qualification pending",
            "jax_imported":"jax" in sys.modules}
    (artifacts/"report.json").write_text(json.dumps(report,indent=2)+'\n')
    binary.unlink()
    print({"passed":report["passed"],"passed_fields":sum(r["passed"] for r in records),"fields":len(records),"wall_seconds":report["wall_seconds"]})
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
