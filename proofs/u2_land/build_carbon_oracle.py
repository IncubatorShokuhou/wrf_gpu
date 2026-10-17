"""Original CARBON/CO2FLUX multi-step REAL4 references, actual MODIS MPTABLE parameters."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_snow_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();routines={k:extract(text,k) for k in ("CARBON","CO2FLUX")};body="\n".join(routines.values())
    fields=set(re.findall(r"parameters%(\w+)",body,re.I))
    integers={"NROOT","ISWATER","ISBARREN","ISICE","EBLFOREST"}
    definitions="\n".join(("logical" if k.lower()=="urban_flag" else "integer" if k.upper() in integers else "real")+" :: "+k+("(4)" if k.upper()=="SMCMAX" else "") for k in sorted(fields))
    (build/"reference.f90").write_text("module reference\nimplicit none\ntype noahmp_parameters\n"+definitions+"\nend type\ncontains\n"+body+"\nend module\n")
    keys=("SLA","FOLNMX","ARM","RMF25","RMR25","RMS25","WDPOOL","WRRAT","LTOVRC","TDLEF","DILEFW","DILEFC","FRAGR","TMIN","MRP")
    (build/"driver.f90").write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: n,i,j,veg,nstep
real :: dt,tv,psn,btran,foln,lai,sai,lfmass,rtmass,stmass,wood,stblcp,fastcp,gpp,npp,nee,autors,heters,totsc,totlb
real :: zsoil(4),dz(-2:4),stc(-2:4),smc(4)
p%ISWATER=17;p%ISBARREN=16;p%ISICE=15;p%EBLFOREST=2
zsoil=[-.1,-.4,-1.,-2.];dz=[0.,0.,0.,.1,.3,.6,1.]
read(*,*)n,nstep
do i=1,n
read(*,*)veg,dt,tv,psn,btran,foln,lai,sai,lfmass,rtmass,stmass,wood,stblcp,fastcp
read(*,*)'''+",".join("p%"+k for k in keys)+'''
read(*,*)p%NROOT,p%SMCMAX,smc,stc
p%urban_flag=veg==13
do j=1,nstep
call CARBON(p,3,4,veg,dt,zsoil,dz,stc,smc,tv,tv,psn,foln,btran,200.,.8,merge(1.,0.,tv>p%TMIN),280.,1,.5,1,1,lfmass,rtmass,stmass,wood,stblcp,fastcp,gpp,npp,nee,autors,heters,totsc,totlb,lai,sai)
write(*,'(15(ES25.16E3,1X))')lfmass,rtmass,stmass,wood,stblcp,fastcp,lai,sai,gpp,npp,nee,autors,heters,totsc,totlb
enddo
enddo
end program
''')
    fc=shutil.which("gfortran") or "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    flags=["-O0","-ffree-form","-ffree-line-length-none","-ffp-contract=off","-finit-real=zero","-finit-integer=0","-finit-logical=false"]
    subprocess.run([fc,*flags,"reference.f90","driver.f90","-o","oracle.exe"],cwd=build,check=True,timeout=120)
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    wrf_root=next(p for p in source.parents if (p/"run/MPTABLE.TBL").exists());table=wrf_root/"run/MPTABLE.TBL"
    params=_parse_mptable_block(table.read_text(),"noahmp_modis_parameters")
    cases=[]
    for veg in (2,5,10,12,13,15,16,17):
        for tv,psn,btran,dt in ((260.,0.,.01,18.),(280.,5.,.5,18.),(300.,15.,1.,1800.),(310.,0.,.01,86400.)):
            parameter={k:float(params[k][veg-1]) for k in keys}
            no_veg=veg in (13,15,16,17)
            lai=0. if no_veg else 2.;sai=0. if no_veg else .2
            state=[lai*(1000./max(parameter["SLA"],1.)),500.,sai*(1000./3.),500.,1000.,1000.]
            if no_veg:state=[0.]*6
            cases.append(dict(veg=veg,dt=dt,tv=tv,psn=psn,btran=btran,foln=1.,lai=lai,sai=sai,state=state,parameters=parameter,
                              nroot=int(params["NROOT"][veg-1]),smcmax=[.45]*4,smc=[.2]*4,stc=[tv]*7))
    nstep=4
    def run(rows):
        lines=[f"{len(rows)} {nstep}"]
        for c in rows:
            lines += [" ".join(map(str,[c[k] for k in ("veg","dt","tv","psn","btran","foln","lai","sai")]+c["state"])),
                      " ".join(str(c["parameters"][k]) for k in keys)," ".join(map(str,[c["nroot"],*c["smcmax"],*c["smc"],*c["stc"]]))]
        r=subprocess.run([str((build/"oracle.exe").resolve())],input="\n".join(lines)+"\n",text=True,capture_output=True,check=True,timeout=30)
        out=np.fromstring(r.stdout,sep=" ",dtype=np.float32).reshape(len(rows),nstep,15);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ("dt","tv","psn","btran","foln","lai","sai","state","smc","stc"):
            for index in (range(len(c[key])) if isinstance(c[key],list) else (None,)):
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
        routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,cases=cases,parameter_keys=list(keys),steps=nstep,ulp_variants=len(variants),
        nominal_tolerance=dict(atol=1.e-6,rtol=3.e-6),scope="32constructed MODIS forest/grass/crop/urban/ice/barren/water,cold/hot/long-step/zeroassimilation/water-stress;4steps each;originalCARBON+CO2FLUX;actualMPTABLEparameters"),indent=1)+"\n")


if __name__=="__main__":main()
