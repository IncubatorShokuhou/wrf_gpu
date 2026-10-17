"""Reproduce undefined KEPS inputs from unchanged private WRF 4.7.1 sources.

CPU only. No build or output is written to the pristine tree. Compiler poisoning
is diagnostic instrumentation, never a numeric truth fixture or a source fix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


COMMON = """
 use module_bl_keps
 implicit none
 integer, parameter :: n=12, m=n+1
 real :: dz(m),z(m),th(m),q(m),te(m),ref(m),ns(m),grad(m),pra(m)
 real :: rho(m),rad(m),pi(m),p8w(m),sh(m),ri(m),prast(m)
 real :: zi,pm,ph,pe,ds
 integer :: k, kp
 dz=100.; z=0.; th=300.; q=.008; te=1.; ref=300.; pra=1.
 rho=1.; rad=0.; pi=1.; sh=.01; ns=-.001
 do k=1,m
   z(k)=(k-1)*100.; th(k)=300.+(k-1)*.5; p8w(k)=100000.-1000.*(k-1)
 enddo
"""
PROBES = {
    "get_pblh_ttke": COMMON + """
 call get_pblh(1,n,zi,th(1:n),te(1:n),z,dz(1:n),q(1:n),kp)
 print *, 'UNEXPECTED_RETURN',zi,kp
 end
""",
    "surface_radsum": COMMON + """
 zi=400.; kp=4; ds=.001
 call surface_bl_pra_ri(1,m,1,n,dz,z,rho,9.81,1004.5,.1,.01,rad,pi,p8w, &
                       th,q,sh,ns,ri,.1,1.,1.,zi,.3,kp,pra,prast,pm,ph,pe,ds)
 print *, 'UNEXPECTED_RETURN',pra(1),pm,ph,pe
 end
""",
    "nsquare_top_gradient": COMMON + """
 grad=-98765.
 call c_nsquare(pra,.3,.01,.1,300.,.01,pi,9.81,1,m,1,n,th,ref,dz,ns,grad)
 print *, 'TOP_GRADIENT',grad(n),'TOP_NSQUARE',ns(n)
 if(grad(n)/=-98765. .or. ns(n)/=0.) stop 2
 end
""",
}

EEPS_PROBE = """program probe
 use module_bl_eeps
 implicit none
 integer,parameter :: n=44
 real :: u(1,n),v(1,n),t(1,n),q(1,n),qc(1,n),qi(1,n),zero(1,n)
 real :: p(1,n),z(1,n),zw(1,n+1),rho(1,n),rad(1,n),te(1,n),ep(1,n)
 real :: kh(1,n+1),km(1,n+1),pblh(1),theta
 integer :: k,kp(1)
 u=8.;v=2.;q=.008;qc=0.;qi=0.;zero=0.;rad=0.;te=.1;ep=.001
 do k=1,n
   z(1,k)=(n-k)*100.+50.
   zw(1,k)=(n+1-k)*100.
   p(1,k)=100000.*exp(-z(1,k)/8000.)
   theta=290.+.001*z(1,k)
   if(z(1,k)>1000.) theta=theta+12.
   t(1,k)=theta*(p(1,k)/100000.)**(287./1004.5)
   rho(1,k)=p(1,k)/(287.*t(1,k)*(1.+(461.6/287.-1.)*q(1,k)))
   if(z(1,k)>=400. .and. z(1,k)<=900.) then
     qc(1,k)=.0005
     rad(1,k)=-.0001
   endif
 enddo
 zw(1,n+1)=0.
 call eeps2d(u,v,t,q,qc,qi,zero,zero,zero,p,z,zw,rho,rad,te,ep, &
             [100000.],[.3],[-.02],[sqrt(68.)],[1.],[.01],[50.],[.00005],[292.], &
             18.,3000.,3,kh,km,pblh,kp,1,n)
 print *, 'UNEXPECTED_RETURN',pblh,kp
end
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wrf-root", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    ap.add_argument("--eeps", action="store_true", help="also run the active-cloud EEPS trap probe")
    args = ap.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    src = args.wrf_root / "phys/module_bl_keps.F"
    staged = args.build / src.name
    shutil.copyfile(src, staged)
    assert sha(src) == sha(staged)
    flags = ["-ffree-form", "-ffree-line-length-none", "-O0", "-g", "-fcheck=all",
             "-finit-real=snan", "-ffpe-trap=invalid,zero,overflow"]
    def run(cmd):
        r = subprocess.run(cmd, cwd=args.build, capture_output=True, text=True, timeout=30)
        return {"command": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    build = run([args.compiler, *flags, "-c", str(staged), "-o", "keps.o"])
    assert build["returncode"] == 0, build
    results = {"source": str(src), "source_sha256": sha(src), "staged_sha256": sha(staged),
               "compiler": run([args.compiler, "--version"])["stdout"].splitlines()[0],
               "flags": flags, "build": build, "probes": {}}
    for name, body in PROBES.items():
        (args.build / f"{name}.f90").write_text("program probe\n" + body)
        compiled = run([args.compiler, *flags, f"{name}.f90", "keps.o", "-o", name])
        assert compiled["returncode"] == 0, compiled
        result = run([str(args.build / name)])
        if name == "nsquare_top_gradient":
            assert result["returncode"] == 0 and "-98765" in result["stdout"], result
        else:
            assert result["returncode"] == -8 and "SIGFPE" in result["stderr"], result
            assert name.split("_")[0] in result["stderr"].lower(), result
        results["probes"][name] = result
    if args.eeps:
        for rel in ("share/module_model_constants.F", "phys/module_bl_eepsilon.F"):
            original = args.wrf_root / rel
            dest = args.build / original.name
            shutil.copyfile(original, dest)
            results.setdefault("additional_sources", {})[rel] = sha(original)
            assert sha(original) == sha(dest)
            compiled = run([args.compiler, *flags, "-c", str(dest), "-o", dest.stem + ".o"])
            assert compiled["returncode"] == 0, compiled
        (args.build / "eeps_radsum.f90").write_text(EEPS_PROBE)
        compiled = run([args.compiler, *flags, "eeps_radsum.f90", "module_bl_eepsilon.o",
                        "module_model_constants.o", "-o", "eeps_radsum"])
        assert compiled["returncode"] == 0, compiled
        result = run([str(args.build / "eeps_radsum")])
        assert result["returncode"] == -8 and "eeps2d" in result["stderr"].lower(), result
        results["probes"]["eeps_radsum"] = result
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"source_sha256": results["source_sha256"],
                      "probe_returncodes": {k: v["returncode"] for k, v in results["probes"].items()}}))


if __name__ == "__main__":
    main()
