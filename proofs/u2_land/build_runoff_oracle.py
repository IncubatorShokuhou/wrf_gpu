"""Original VIC/XAJ/dynamic VIC family; input-ULP and uninitialized-YD sensitivity."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_crop_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();names=('COMPUTE_VIC_SURFRUNOFF','COMPUTE_XAJ_SURFRUNOFF','DYNAMIC_VIC','RR1','RR2','PHILIP_INFIL','GREEN_AMPT_INFIL','SMITH_PARLANGE_INFIL','WDFCND2')
    routines={k:extract(text,k) for k in names};body='\n'.join(routines.values())
    fields=sorted(set(k.upper() for k in re.findall(r'parameters%(\w+)',body,re.I)))
    layers={'SMCMAX','SMCREF','SMCWLT','BEXP','DKSAT','DWSAT'}
    definitions='\n'.join('real :: '+k+('(4)' if k in layers else '') for k in fields)
    (build/'reference.f90').write_text('module reference\nimplicit none\ninteger :: OPT_INFDV\ntype noahmp_parameters\n'+definitions+'\nend type\ncontains\n'+body+'\nend module\n')
    (build/'driver.f90').write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: n,i,j,run
real :: smc(4),sh2o(4),sice(4),zsoil(4),fcr(4),qin,dt,facc,asat,runsrf,pddum
zsoil=[-.1,-.4,-1.,-2.]
read(*,*)n
do i=1,n
read(*,*)run,OPT_INFDV,dt,qin,smc,sh2o,fcr
read(*,*)'''+','.join('p%'+k for k in fields)+'''
sice=smc-sh2o;facc=1.e-6
do j=1,4
if(run==6)call COMPUTE_VIC_SURFRUNOFF(p,dt,4,smc,zsoil,qin,asat,runsrf,pddum)
if(run==7)call COMPUTE_XAJ_SURFRUNOFF(p,dt,fcr,4,smc,zsoil,qin,runsrf,pddum)
if(run==8)call DYNAMIC_VIC(p,dt,smc,sh2o,sice,maxval(sice),4,zsoil,qin,facc,pddum,runsrf)
write(*,'(3(ES25.16E3,1X))')pddum,runsrf,facc
enddo
enddo
end program
''')
    fc=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-ffree-form','-ffree-line-length-none','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    subprocess.run([fc,*flags,'reference.f90','driver.f90','-o','oracle.exe'],cwd=build,check=True,timeout=120)
    sensitivity=[v if v!='-finit-real=zero' else '-finit-real=nan' for v in flags]
    subprocess.run([fc,*sensitivity,'reference.f90','driver.f90','-o','sensitivity.exe'],cwd=build,check=True,timeout=120)
    wrf_root=next(p for p in source.parents if (p/'run/SOILPARM.TBL').exists());table=wrf_root/'run/SOILPARM.TBL'
    lines=table.read_text().splitlines();count=int(lines[2].split(',')[0])
    columns=('BEXP','SMCDRY','F1','SMCMAX','SMCREF','PSISAT','DKSAT','DWSAT','SMCWLT','QUARTZ','BVIC','AXAJ','BXAJ','XXAJ','BDVIC','BBVIC','GDVIC')
    soils={int(row.split(',')[0]):dict(zip(columns,map(float,row.split(',')[1:18]))) for row in lines[3:3+count]}
    cases=[]
    for run,infdv in ((6,0),(7,0),(8,1),(8,2),(8,3)):
        for soil,moist,ice,qin,dt in ((3,.2,0.,0.,18.),(3,.12,0.,1.e-6,18.),(4,.3,0.,1.e-4,1800.),(6,.3,0.,.01,18.),
                (7,.1,.2,1.e-5,18.),(8,.05,.3,.001,18.),(12,.005,.4,0.,18.),(3,.434,0.,1.e-4,18.)):
            p={k:[soils[soil][k]]*4 if k in layers else soils[soil][k] for k in fields}
            smc=[min(moist+ice,soils[soil]['SMCMAX'])]*4;sh2o=[min(moist,v) for v in smc]
            frozen=np.asarray(smc,np.float32)-np.asarray(sh2o,np.float32)
            ratio=np.minimum(np.float32(1),frozen/np.float32(soils[soil]['SMCMAX']))
            fcr=np.maximum(np.float32(0),np.exp(np.float32(-4)*(np.float32(1)-ratio))-np.exp(np.float32(-4)))/(np.float32(1)-np.exp(np.float32(-4)))
            cases.append(dict(opt_run=run,opt_infdv=infdv,dt=dt,qinsur=qin,smc=smc,sh2o=sh2o,fcr=fcr.tolist(),parameters=p,soil=soil))
    def run(rows,executable='oracle.exe'):
        lines=[str(len(rows))]
        for c in rows:
            lines+=[' '.join(map(str,[c['opt_run'],c['opt_infdv'],c['dt'],c['qinsur'],*c['smc'],*c['sh2o'],*c['fcr']])),
                ' '.join(str(v) for k in fields for v in (c['parameters'][k] if k in layers else [c['parameters'][k]]))]
        r=subprocess.run([str((build/executable).resolve())],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=60)
        return np.fromstring(r.stdout,sep=' ',dtype=np.float32).reshape(len(rows),4,3)
    expected=run(cases);assert np.isfinite(expected).all()
    other=run(cases,'sensitivity.exe')
    variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ('dt','qinsur','smc','sh2o','fcr'):
            for k in (range(4) if isinstance(c[key],list) else (None,)):
                for direction in (-np.inf,np.inf):
                    v=c.copy();value=c[key][k] if k is not None else c[key]
                    changed=float(np.nextafter(np.float32(value),np.float32(direction)))
                    if k is None:v[key]=changed
                    else:v[key]=c[key].copy();v[key][k]=changed
                    variants.append(v);owner.append(i)
    results=run(variants);assert np.isfinite(results).all() and np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit='f52c197ed39d12e087d02c50f412d90d418f6186',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},
        table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,
        cases=cases,parameter_fields=fields,ulp_variants=len(variants),steps=4,nominal_tolerance=dict(atol=1.e-8,rtol=3.e-6),
        initialization_sensitivity=dict(flags=sensitivity,nonfinite=int((~np.isfinite(other)).sum()),different=int((other.view(np.uint32)!=expected.view(np.uint32)).sum()),
            max_abs=float(np.nanmax(np.abs(other.astype(float)-expected))),alternative=other.tolist()),
        scope='40 constructed runoff columns actualSTASsoil tables,3runoff+3infiltration choices,8wet/frozen/rain regimes,4calls with persistentFACC'),indent=1)+'\n')


if __name__=='__main__':main()
