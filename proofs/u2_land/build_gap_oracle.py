"""Verbatim original TWOSTREAM geometry block with ALBEDO's zero gap initialization."""
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();block=text[text.index("     VAI_SPREAD = VAI"):text.index("! calculate two-stream parameters OMEGA")]
    driver='''program oracle
implicit none
type parameter_type
real :: rc,hvt,hvb
end type
type(parameter_type) :: parameters
integer :: n,i,opt_rad
real,parameter :: pai=3.14159265
real :: vai_spread,vai,cosz,fveg,denfveg,hd,bb,thetap,bgap,fa,newvai,wgap,gap,kopen
read(*,*) n
do i=1,n
read(*,*) opt_rad,cosz,vai,fveg,parameters%rc,parameters%hvt,parameters%hvb
bgap=0.;wgap=0.
'''+block+'''write(*,'(4(ES25.16E3,1X))') gap,kopen,bgap,wgap
enddo
end program
'''
    (build/"driver.f90").write_text(driver)
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    # O2 rewrites COS(ATAN(x)) to reciprocal-SQRT, eliminating the rounded
    # THETAP assignment even with protect-parens/no-frontend-optimize. O0 keeps
    # both original libm calls and the REAL4 assignment; see LAND13 controls.
    flags=["-O0","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    cases=[dict(opt_rad=opt,cosz=z,vai=v,fveg=f,rc=1.2,hvt=20.,hvb=8.)
           for opt in (1,2,3) for z in (.01,.3,.9) for v in (0.,.04,2.,6.) for f in (.05,.5,.95)]
    def run(rows):
        text=str(len(rows))+"\n"+"\n".join(" ".join(str(c[k]) for k in ("opt_rad","cosz","vai","fveg","rc","hvt","hvb")) for c in rows)+"\n"
        result=subprocess.run([str((build/"oracle.exe").resolve())],input=text,text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(result.stdout,sep=" ",dtype=np.float32).reshape(len(rows),4);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for k in ("cosz","vai","fveg","rc","hvt","hvb"):
            for d in (-np.inf,np.inf):
                row=c.copy();row[k]=float(np.nextafter(np.float32(c[k]),np.float32(d)))
                variants.append(row);owner.append(i)
    results=run(variants);assert np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit="f52c197ed39d12e087d02c50f412d90d418f6186",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),block_sha256=hashlib.sha256(block.encode()).hexdigest(),flags=flags,
        created_utc=datetime.now(timezone.utc).isoformat(),cases=cases,variables_out=["GAP","KOPEN","BGAP","WGAP"],ulp_variants=len(variants),
        nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6),scope="108constructed daytime solar/canopy/vegetation columns; TWOSTREAM opt_rad1/2/3; ALBEDO caller initializes BGAP/WGAP0"),indent=1)+"\n")


if __name__=="__main__":main()
