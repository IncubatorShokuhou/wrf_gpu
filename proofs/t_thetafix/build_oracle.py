"""Build eight original WRF 4.7.1 callers in a private lane directory.

Original module bodies are copied byte for byte. The only stubs are WRF logging
and fatal-error plumbing. Driver assignments are extracted from pristine WRF;
its scheme calls follow module_microphysics_driver.F keyword mappings.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WRF = ROOT / "data/wrf_pristine/WRF"
FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
FLAGS = ["-O1", "-cpp", "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off",
         "-fno-range-check", "-finit-real=zero", "-finit-integer=0", "-finit-logical=false"]
DIMS = "ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=n,its=1,ite=1,jts=1,jte=1,kts=1,kte=n"
COMMON = "th=th_phy,q=qv,den=rho,pii=pii,p=p,delz=dz,delt=dt,g=g,cpd=cp,cpv=cpv,rd=r_d,rv=r_v,t0c=svpt0,ep1=ep_1,ep2=ep_2,qmin=epsilon,xls=xls,xlv0=xlv,xlf0=xlf,den0=rhoair0,denr=rhowater,cliq=cliq,cice=cice,psat=psat,rain=rain,rainncv=rainv,snow=snow,snowncv=snowv,sr=sr,has_reqc=0,has_reqi=0,has_reqs=0,re_cloud=reqc,re_ice=reqi,re_snow=reqs"


def build(out, precision="fp32"):
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists()
    out.mkdir(parents=True, exist_ok=True)
    paths = ["share/module_model_constants.F", "frame/libmassv.F", "phys/ccpp_kind_types.F",
             "phys/physics_mmm/module_libmassv.F90", "phys/physics_mmm/mp_radar.F90",
             "phys/physics_mmm/mp_wsm6.F90", "phys/physics_mmm/mp_wsm6_effectRad.F90",
             "phys/module_mp_radar.F"]
    paths += [f"phys/module_mp_{n}.F" for n in ("kessler", "wsm3", "wsm5", "wsm6",
                                                "morr_two_moment", "sbu_ylin", "wdm5", "gsfcgce")]
    commands, sources = [], {}
    stub = ROOT / "proofs/v060/oracle/morrison_stub_modules.f90"
    shutil.copyfile(stub, out / stub.name)
    compile_paths = [out / stub.name]
    for rel in paths:
        src = WRF / rel
        dst = out / src.name
        shutil.copyfile(src, dst)
        sources[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
        assert dst.read_bytes() == src.read_bytes()
        compile_paths.append(dst)
    lines = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text().splitlines()
    prep = next(l for l in lines if "th_phy(i,k,j) = (t_new(i,k,j) + t0) / (1. + (R_v/R_d) * qv(i,k,j))" in l)
    idx = next(i for i, l in enumerate(lines) if "t_new(i,k,j) =  h_diabatic(i,k,j)*(1." in l)
    finish = "\n".join(lines[idx:idx+3])
    calls = [
        "case(1)\ncall kessler(t=th_phy,qv=qv,qc=qc,qr=qr,rho=rho,pii=pii,dt_in=dt,z=z,xlv=xlv,cp=cp,ep2=ep_2,svp1=.6112,svp2=17.67,svp3=29.65,svpt0=svpt0,rhowater=rhowater,dz8w=dz,rainnc=rain,rainncv=rainv," + DIMS + ")",
        "case(3)\ncall wsm3(qci=qc,qrs=qr,w=w," + COMMON + "," + DIMS + ")",
        "case(4)\ncall wsm5(qc=qc,qr=qr,qi=qi,qs=qs,refl_10cm=refl,diagflag=.false.,do_radar_ref=0," + COMMON + "," + DIMS + ")",
        "case(6)\ncall wsm6(qc=qc,qr=qr,qi=qi,qs=qs,qg=qg,graupel=graupel,graupelncv=graupelv,refl_10cm=refl,diagflag=.false.,do_radar_ref=0,re_qc_bg=re_qc_bg,re_qi_bg=re_qi_bg,re_qs_bg=re_qs_bg,re_qc_max=re_qc_max,re_qi_max=re_qi_max,re_qs_max=re_qs_max,errmsg=errmsg,errflg=errflg," + COMMON + "," + DIMS + ")\nif(errflg/=0) stop 6",
        "case(10)\ncall mp_morr_two_moment(step,th_phy,qv,qc,qr,qi,qs,qg,ni,ns,nr,ng,rho,pii,p,dt,dz,ht,w,rain,rainv,sr,snow,snowv,graupel,graupelv,refl,.false.,0,qrcu,qscu,qicu," + DIMS + ")",
        "case(13)\ncall sbu_ylin(th=th_phy,qv=qv,ql=qc,qr=qr,qi=qi,qs=qs,ri3d=ri,rho=rho,pii=pii,p=p,dt_in=dt,z=z,ht=ht,dz8w=dz,rainnc=rain,rainncv=rainv," + DIMS + ")",
        # These are continued hour-six columns, not WRF's first-step CCN reset.
        "case(14)\ncall wdm5(qc=qc,qr=qr,qi=qi,qs=qs,nn=nn,nc=nc,nr=nr,ccn0=1.e8,refl_10cm=refl,diagflag=.false.,do_radar_ref=0,itimestep=1200+step," + COMMON + "," + DIMS + ")",
        "case(97)\ncall gsfcgce(th=th_phy,qv=qv,ql=qc,qr=qr,qi=qi,qs=qs,qg=qg,f_qg=.true.,rho=rho,pii=pii,p=p,dt_in=dt,z=z,ht=ht,dz8w=dz,grav=g,rhowater=rhowater,rhosnow=rhosnow,itimestep=step,rainnc=rain,rainncv=rainv,snownc=snow,snowncv=snowv,graupelnc=graupel,graupelncv=graupelv,sr=sr,refl_10cm=refl,diagflag=.false.,do_radar_ref=0,ihail=0,ice2=0," + DIMS + ")",
    ]
    driver = (ROOT / "proofs/t_thetafix/caller.f90.in").read_text().replace("@PREP@", prep).replace("@FINISH@", finish).replace("@CALLS@", "\n".join(calls))
    (out / "caller.f90").write_text(driver)
    compile_paths.append(out / "caller.f90")
    for path in compile_paths:
        flags = FLAGS if path.name != "libmassv.F" else [
            f for f in FLAGS if f not in ("-ffree-form", "-ffree-line-length-none")
        ] + ["-ffixed-form", "-ffixed-line-length-none"]
        if precision == "fp64":
            flags = [*flags,"-fdefault-real-8","-fdefault-double-8","-DDOUBLE_PRECISION"]
        cmd = [FC, *flags, "-c", path.name]
        subprocess.run(cmd, cwd=out, check=True, timeout=120)
        commands.append(cmd)
    cmd = [FC, "-o", "caller", *[p.stem + ".o" for p in compile_paths]]
    subprocess.run(cmd, cwd=out, check=True, timeout=60)
    commands.append(cmd)
    sources["dyn_em/module_big_step_utilities_em.F"] = hashlib.sha256((WRF / "dyn_em/module_big_step_utilities_em.F").read_bytes()).hexdigest()
    sources["phys/module_microphysics_driver.F"] = hashlib.sha256((WRF / "phys/module_microphysics_driver.F").read_bytes()).hexdigest()
    (out / "build.json").write_text(json.dumps(dict(sources=sources, commands=commands,
        driver_sha256=hashlib.sha256(driver.encode()).hexdigest(), prep=prep, finish=finish,
        precision=precision, carry="REAL4 at each caller boundary, matching existing dispatcher island",
        scope="two MP caller steps; geometry and Exner held at port caller inputs; not full forecast", platform="cpu"), indent=2)+"\n")
    print("built pristine eight-scheme caller", flush=True)


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve(),sys.argv[2] if len(sys.argv)>2 else "fp32")
