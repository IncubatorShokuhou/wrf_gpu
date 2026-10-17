"""Complete original SOILWATER 3/4 and INF1/2, with original fine-step caller cadence."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    runs=tuple(map(int,sys.argv[4:])) or (3,4)
    columns_out=15 if 1 in runs else 12 if 2 in runs else 11
    text=source.read_text();names=("SOILWATER","SRT","SSTEP","INFIL","WDFCND1","WDFCND2","ROSR12","ZWTEQ","GROUNDWATER")
    routines={k:extract(text,k) for k in names};body="\n".join(routines.values())
    fields=set(re.findall(r"parameters%(\w+)",body,re.I));arrays=set(re.findall(r"parameters%(\w+)\(",body,re.I))
    definitions="\n".join(("logical" if name.lower()=="urban_flag" else "real")+" :: "+name+("(4)" if name in arrays else "") for name in sorted(fields))
    (build/"reference.f90").write_text('''module reference
implicit none
integer :: opt_run,opt_inf,opt_tdrn=0,opt_infdv=0
type noahmp_parameters
'''+definitions+"\nend type\ncontains\n"+body+"\nend module\n")
    calls=set(re.findall(r"\bCALL\s+(\w+)",body,re.I));stubs=calls-set(names)
    # These other-option routines are unreachable for the declared run3/4,
    # tile-drain0 scope. Abort if any one is accidentally entered.
    stubtext="\n".join("subroutine "+name+"\nstop 99\nend subroutine\n" for name in sorted(stubs))
    (build/"driver.f90").write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: i,n,urban
real :: dt,qinsur,qseva,sh2o(4),smc(4),sice(4),etrani(4),zsoil(4),dz(4),zwt,smcwtd,deeprech,runsrf,qdrain,runsub,wcnd(4),fcrmax,qtldrn,wa,wt,qin,qdis,stc(4)
read(*,*) p%bexp
read(*,*) p%smcmax
read(*,*) p%smcwlt
read(*,*) p%dksat
read(*,*) p%dwsat
read(*,*) p%psisat
read(*,*) p%kdt,p%frzx,p%slope
read(*,*) p%fsatmx,p%timean
read(*,*) zsoil
dz(1)=-zsoil(1);dz(2:4)=zsoil(1:3)-zsoil(2:4)
read(*,*) n
do i=1,n
read(*,*) opt_run,opt_inf,urban,dt,qinsur,qseva,sh2o,smc,etrani
p%urban_flag=urban==1
sice=max(0.,smc-sh2o);zwt=2.5;wa=4900.;wt=4900.;smcwtd=.2;deeprech=0.;runsub=0.;qtldrn=0.;stc=270.;qin=0.
call SOILWATER(p,4,0,dt,zsoil,dz,qinsur,qseva,etrani,sice,1,1,0.,3000.,sh2o,smc,zwt,14,smcwtd,deeprech,runsrf,qdrain,runsub,wcnd,fcrmax,qtldrn)
if(opt_run==1)then
call GROUNDWATER(p,0,4,dt,sice,zsoil,stc,wcnd,fcrmax,1,1,sh2o,zwt,wa,wt,qin,qdis)
runsub=qdis
smc=sh2o+sice
endif
'''+("write(*,'(15(ES25.16E3,1X))') sh2o,smc,runsrf,qdrain,runsub,zwt,wa,wt,qin\n" if columns_out==15 else "write(*,'(12(ES25.16E3,1X))') sh2o,smc,runsrf,qdrain,runsub,zwt\n" if columns_out==12 else "write(*,'(11(ES25.16E3,1X))') sh2o,smc,runsrf,qdrain,runsub\n")+'''
enddo
end program
'''+stubtext)
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O0","-cpp","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    p=dict(bexp=[4.5,5.,4.,5.5],smcmax=[.45,.47,.45,.43],smcwlt=[.1,.12,.1,.08],dksat=[1.e-5,2.e-6,5.e-6,1.e-6],dwsat=[1.e-5,1.2e-5,1.5e-5,1.e-5],psisat=[.35,.4,.3,.45],kdt=3.,frzx=.15,slope=.1,fsatmx=.38,timean=10.5)
    zsoil=[-.1,-.4,-1.,-2.]
    cases=[]
    for run in runs:
        for inf in (1,2):
            for qin,sh,ice in ((0.,.2,0.),(1.e-6,.12,0.),(1.e-4,.3,0.),(1.e-2,.3,0.),(1.e-5,.1,.2),(1.e-3,.05,.3),(0.,.005,.4),(1.e-4,.44,0.)):
                cases.append(dict(opt_run=run,opt_inf=inf,urban=0,dt=18.,qinsur=qin,qseva=1.e-8,
                    sh2o=[sh,min(sh,.47),min(sh,.45),min(sh,.43)],smc=[min(sh+ice,v) for v in p["smcmax"]],etrani=[1.e-8,2.e-8,1.e-8,0.]))
    def run(rows):
        lines=[" ".join(map(str,p[k])) for k in ("bexp","smcmax","smcwlt","dksat","dwsat","psisat")]
        lines += [f'{p["kdt"]} {p["frzx"]} {p["slope"]}',f'{p["fsatmx"]} {p["timean"]}'," ".join(map(str,zsoil)),str(len(rows))]
        lines += [" ".join(map(str,[c["opt_run"],c["opt_inf"],c["urban"],c["dt"],c["qinsur"],c["qseva"],*c["sh2o"],*c["smc"],*c["etrani"]])) for c in rows]
        r=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=" ",dtype=np.float32).reshape(len(rows),columns_out);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ("dt","qinsur","qseva","sh2o","smc","etrani"):
            for index in (range(4) if isinstance(c[key],list) else (None,)):
                for d in (-np.inf,np.inf):
                    v=c.copy();value=c[key][index] if index is not None else c[key]
                    changed=float(np.nextafter(np.float32(value),np.float32(d)))
                    if index is None:v[key]=changed
                    else:v[key]=c[key].copy();v[key][index]=changed
                    variants.append(v);owner.append(i)
    results=run(variants);assert np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},flags=flags,created_utc=datetime.now(timezone.utc).isoformat(),
        unreachable_stubs=sorted(stubs),parameters=p,zsoil=zsoil,cases=cases,ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6),
        scope=f"{len(cases)}constructed entireSOILWATER dry/wet/frozen/saturation/largepulse;run{runs} INF1/2 urbanfalse tile0;other-option stubs abort if entered; allprognostic operands independently +/-1ULP"),indent=1)+"\n")


if __name__=="__main__":main()
