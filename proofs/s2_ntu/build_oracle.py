"""Stage the unmodified WRF NTU source and its real initialization/MP caller.

Run only on CPU, from an immutable checkout. Artifacts/builds stay in the lane
directory; the original module is never edited. Floating initialization is a
compiler policy, not a replacement for any WRF statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
VOLUME = "th qv qc qr qi qs qg qh nc nr ni ns ng nh qdcn qtcn qccn qrcn qnin fi fs vi vs vg ai as ag ah i3m".split()
SURFACE = "rainnc rainncv snownc snowncv grapnc grapncv hailnc hailncv sr".split()
NK = 12
CASES = ("dry", "warm_cloud", "warm_rain", "ice", "snow", "graupel", "hail", "mixed",
         "deep_mixed", "evaporation", "ice_supersaturation", "zero_numbers")


def driver():
    declarations = ",".join(VOLUME + ["p", "pb", "ph", "phb", "alt", "pii", "dz", "w"])
    surface = ",".join(SURFACE + ["xland"])
    # Full MP_NTU keyword mapping, matching module_microphysics_driver.F:1586.
    mapping = ", &\n".join(f"{name}={name}" for name in VOLUME + SURFACE)
    mapping += ", &\np=p,dz=dz,w=w,pii=pii,dt_mp=dt,itimestep=step"
    bounds = ", &\nids=1,ide=2,jds=1,jde=2,kds=1,kde=nk+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=nk+1,its=1,ite=1,jts=1,jte=1,kts=1"
    reads = "; ".join(f"read(u) {x}(1,1:nk,1)" for x in VOLUME)
    reads += "; " + "; ".join(f"read(u) {x}" for x in SURFACE)
    writes = "; ".join(f"write(u) {x}(1,1:nk,1)" for x in VOLUME)
    writes += "; " + "; ".join(f"write(u) {x}" for x in SURFACE)
    return f"""program oracle
use module_mp_ntu
implicit none
integer,parameter::nk={NK},ncase={len(CASES)}
real,dimension(1,nk+1,1)::{declarations}
real,dimension(1,1)::{surface}
real::dt,t,rh,qsat,qmass,diam,rhoi,nspec(6),d(6),density(6),mass(6)
integer::c,k,u,step,ccnty,first,last,frozen
character(300)::arg,filename
call get_command_argument(1,arg);read(arg,*)first
call get_command_argument(2,arg);read(arg,*)last
call get_command_argument(3,arg);read(arg,*)ccnty
call get_command_argument(4,arg);read(arg,*)frozen
dt=18.
do c=first,last
  th=0.;qv=0.;qc=0.;qr=0.;qi=0.;qs=0.;qg=0.;qh=0.
  nc=0.;nr=0.;ni=0.;ns=0.;ng=0.;nh=0.
  qdcn=0.;qtcn=0.;qccn=0.;qrcn=0.;qnin=0.
  fi=0.;fs=0.;vi=0.;vs=0.;vg=0.;ai=0.;as=0.;ag=0.;ah=0.;i3m=0.
  rainnc=0.;rainncv=0.;snownc=0.;snowncv=0.;grapnc=0.;grapncv=0.;hailnc=0.;hailncv=0.;sr=0.
  ph=0.;pb=0.;p=0.;w=0.5;dz=250.;xland=1.
  if(mod(c,2)==0)xland=2.
  do k=1,nk+1
    phb(1,k,1)=real(k-1)*250.*9.806
    p(1,k,1)=95000.*exp(-real(k-1)*250./8000.)
    pii(1,k,1)=(p(1,k,1)/100000.)**(287.058/1005.46)
    t=285.-real(k-1)*2.
    if(c>=4.and.c<=7)t=261.-real(k-1)*2.
    if(c==9)t=290.-real(k-1)*5.
    if(c==11)t=250.-real(k-1)*2.
    th(1,k,1)=t/pii(1,k,1)
    rh=0.99
    if(c==1.or.c==10)rh=0.65
    if(c==11)rh=1.08
    ! Bolton is only an input constructor, not an expected-value oracle.
    qsat=0.622*611.2*exp(17.67*(t-273.15)/(t-29.65))/p(1,k,1)
    qv(1,k,1)=rh*qsat
    alt(1,k,1)=287.058*t*(1.+qv(1,k,1))/p(1,k,1)
  enddo
  call ntu_init(phb=phb,ph=ph,p=p,pb=pb,alt=alt,qv=qv,qdcn=qdcn,qtcn=qtcn,qccn=qccn,qrcn=qrcn,qnin=qnin,xland=xland,ccnty=ccnty,restart=.false.{bounds},kte=nk+1)
  d=(/20.e-6,500.e-6,100.e-6,1000.e-6,2000.e-6,5000.e-6/)
  density=(/997.,997.,500.,100.,400.,900./)
  do k=1,nk
    mass=0.
    if(c==2)mass(1)=1.e-4
    if(c==3)mass(1:2)=(/1.e-4,2.e-4/)
    if(c>=4.and.c<=7)mass(c-1)=2.e-4
    if(c>=8)mass=(/1.e-4,2.e-4,3.e-5,1.e-4,1.e-4,5.e-5/)
    nspec=mass/(3.141592653589793* d**3*density/6.)
    qc(1,k,1)=mass(1);qr(1,k,1)=mass(2);qi(1,k,1)=mass(3)
    qs(1,k,1)=mass(4);qg(1,k,1)=mass(5);qh(1,k,1)=mass(6)
    nc(1,k,1)=nspec(1);nr(1,k,1)=nspec(2);ni(1,k,1)=nspec(3)
    ns(1,k,1)=nspec(4);ng(1,k,1)=nspec(5);nh(1,k,1)=nspec(6)
    vi(1,k,1)=mass(3)/density(3);vs(1,k,1)=mass(4)/density(4);vg(1,k,1)=mass(5)/density(5)
    i3m(1,k,1)=mass(3)*6./(3.141592653589793*density(3))
    fi(1,k,1)=i3m(1,k,1)*0.8;fs(1,k,1)=mass(4)/density(4)*0.8
    ai(1,k,1)=nspec(3)*d(3)**2;as(1,k,1)=nspec(4)*d(4)**2
    ag(1,k,1)=nspec(5)*d(5)**2;ah(1,k,1)=nspec(6)*d(6)**2
    if(c>=8)then
      qccn(1,k,1)=qtcn(1,k,1)*0.05;qrcn(1,k,1)=qtcn(1,k,1)*0.02
      qdcn(1,k,1)=qtcn(1,k,1)-qccn(1,k,1)-qrcn(1,k,1)
    endif
  enddo
  if(c==12)then
    nc=0.;nr=0.;ni=0.;ns=0.;ng=0.;nh=0.
  endif
  if(frozen==1)then
    write(filename,'(a,i2.2,a)')'input_',c,'.bin'
    open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='old')
    {reads}
    read(u)p(1,1:nk,1);read(u)dz(1,1:nk,1);read(u)pii(1,1:nk,1);read(u)w(1,:,1);read(u)dt
    close(u)
  endif
  write(filename,'(a,i2.2,a)')'case_',c,'.bin'
  open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='replace')
  {writes}
  write(u)p(1,1:nk,1);write(u)dz(1,1:nk,1);write(u)pii(1,1:nk,1);write(u)w(1,:,1);write(u)dt
  do step=1,2
    call mp_ntu({mapping}{bounds},kte=nk)
    {writes}
  enddo
  close(u)
enddo
end program oracle
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--kind", choices=("r4", "r8"), default="r4")
    parser.add_argument("--init", choices=("zero", "snan", "inf"), default="zero")
    parser.add_argument("--single", type=int)
    parser.add_argument("--ccnty", type=int, default=2, choices=range(1,5))
    parser.add_argument("--input-npz", type=Path, help="Freeze MP inputs when comparing compiler policies")
    parser.add_argument("--tables-npz", type=Path, help="Freeze original AERO_CONST save arrays across compiler policies")
    parser.add_argument("--fma",action="store_true",help="Original optimized native-FMA control; frozen inputs recommended")
    parser.add_argument("--dt",type=float,help="Explicit alternative timestep after reading frozen inputs")
    parser.add_argument("--retain-build",action="store_true",help="Control-sweep caller owns exact build cleanup")
    args = parser.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert not Path("/tmp/wrf_gpu2_quiet").exists()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2] / "data/wrf_pristine/WRF/phys/module_mp_ntu.F"
    staged = out / "module_mp_ntu.F90"
    # Only preprocessor lines are removed; preserve the production REAL4 branch.
    subprocess.run([FC, "-E", "-P", "-cpp", "-ffree-form", str(source), "-o", str(staged)], check=True)
    caller=driver()
    insertion=""
    if args.tables_npz:
        from aerosol_tables import REAL_FIELDS,DOUBLE_FIELDS,INT_FIELDS
        names=list(REAL_FIELDS)+list(DOUBLE_FIELDS)+list(INT_FIELDS)
        setter="\nsubroutine set_frozen_aerosol_tables()\ninteger::u\nopen(newunit=u,file='frozen_tables.bin',access='stream',form='unformatted',status='old')\n"
        setter+='\n'.join('read(u) '+name for name in names)+"\nclose(u)\nend subroutine set_frozen_aerosol_tables\n"
        text=staged.read_text();match=__import__('re').search(r'^\s*END MODULE MODULE_MP_NTU\s*$',text,__import__('re').I|__import__('re').M);assert match
        staged.write_text(text[:match.start()]+setter+text[match.start():])
        with np.load(args.tables_npz) as tables,(out/'frozen_tables.bin').open('wb') as f:
            for name in names:tables[f'ccnty{args.ccnty}_{name}'].ravel(order='F').tofile(f)
        insertion+='  call set_frozen_aerosol_tables()\n'
    if args.dt is not None:insertion+=f'  dt={args.dt:.17g}\n'
    marker="  write(filename,'(a,i2.2,a)')'case_'"
    assert caller.count(marker)==1
    caller=caller.replace(marker,insertion+marker)
    (out / "error.F90").write_text("""module module_wrf_error
contains
subroutine wrf_error_fatal(message)
character(*),intent(in)::message
print *,message
error stop 1
end subroutine
end module
""")
    (out / "caller.F90").write_text(caller)
    command = [FC,"-ffree-form","-ffree-line-length-none","-O0","-ffp-contract=off",
               "-fno-range-check",f"-finit-real={args.init}","-finit-integer=0","-finit-logical=false",
               "error.F90","module_mp_ntu.F90","caller.F90","-o","oracle"]
    if args.kind == "r8":
        command[1:1] = ["-fdefault-real-8","-fdefault-double-8"]
    if args.fma:command[1:1]=['-O3','-ffp-contract=fast','-march=native']
    if args.fma:
        command.remove('-O0');command.remove('-ffp-contract=off')
    subprocess.run(command, cwd=out, check=True)
    first,last = (args.single,args.single) if args.single else (1,len(CASES))
    dtype = np.dtype("float32" if args.kind == "r4" else "float64")
    if args.input_npz:
        with np.load(args.input_npz) as frozen:
            for c in range(first,last+1):
                arrays = [frozen[f"c{c:02}_s0_{name}"] for name in VOLUME+SURFACE]
                arrays += [frozen[f"c{c:02}_input_{name}"] for name in ["p","dz","pii","w","dt"]]
                assert all(x.dtype == dtype for x in arrays)
                np.concatenate(arrays).tofile(out/f"input_{c:02}.bin")
    with (out / "stdout.log").open("w") as log:
        subprocess.run([str(out/"oracle"),str(first),str(last),str(args.ccnty),str(int(args.input_npz is not None))],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    fields = {}
    width = len(VOLUME)*NK + len(SURFACE)
    for c in range(first,last+1):
        values = np.fromfile(out/f"case_{c:02}.bin", dtype=dtype)
        expected = 3*width+3*NK+(NK+1)+1
        assert values.size == expected, (c,values.size,expected)
        cursor = 0
        for stage in range(3):
            for name in VOLUME:
                fields[f"c{c:02}_s{stage}_{name}"] = values[cursor:cursor+NK].copy();cursor+=NK
            for name in SURFACE:
                fields[f"c{c:02}_s{stage}_{name}"] = values[cursor:cursor+1].copy();cursor+=1
            if stage==0:
                for name,n in [("p",NK),("dz",NK),("pii",NK),("w",NK+1),("dt",1)]:
                    fields[f"c{c:02}_input_{name}"]=values[cursor:cursor+n].copy();cursor+=n
        assert cursor == values.size
    np.savez_compressed(out/"columns.npz",**fields)
    nonfinite={k:int(np.count_nonzero(~np.isfinite(v))) for k,v in fields.items() if not np.isfinite(v).all()}
    manifest = {"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "module_body":"unmodified; cpp production REAL4 branch", "compile":command,
                "kind":args.kind,"init":args.init,"ccnty":args.ccnty,"cases":list(CASES[first-1:last]),
                "frozen_input_npz":None if args.input_npz is None else str(args.input_npz),
                "frozen_tables_npz":None if args.tables_npz is None else str(args.tables_npz),
                "fma":args.fma,"dt_override":args.dt,
                "volume_fields":VOLUME,"surface_fields":SURFACE,"levels":NK,"steps":2,
                "nonfinite":nonfinite,"npz_sha256":hashlib.sha256((out/"columns.npz").read_bytes()).hexdigest(),
                "census":{name:int(sum(np.count_nonzero(fields[f"c{c:02}_s0_{name}"]) for c in range(first,last+1))) for name in VOLUME}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))
    if args.retain_build:
        return
    # Remove exact private disposable build products, preserve only evidence.
    for name in ["error.F90","module_mp_ntu.F90","caller.F90","oracle","module_wrf_error.mod","module_mp_ntu.mod"]:
        (out/name).unlink(missing_ok=True)
    for c in range(first,last+1):
        (out/f"case_{c:02}.bin").unlink()
        (out/f"input_{c:02}.bin").unlink(missing_ok=True)
    (out/'frozen_tables.bin').unlink(missing_ok=True)


if __name__ == "__main__":
    main()
