"""Verbatim Liu crop oracle; every stage and all five actual MPTABLE crops."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
def extract(text,name):
    match=re.search(rf'^[ \t]*SUBROUTINE[ \t]+{name}\b.*?^[ \t]*END SUBROUTINE[ \t]+{name}[ \t]*$',text,re.M|re.S|re.I)
    if match is None:raise ValueError(f'Missing pristine routine {name}')
    return match[0]


SCALARS=('PLTDAY','HSDAY','GDDTBASE','GDDTCUT','GDDS1','GDDS2','GDDS3','GDDS4','GDDS5','Q10MR','FOLN_MX','LEFREEZ','FRA_GR','LFMR25','STMR25','RTMR25','GRAINMR25','BIO2LAI','MRP')
ARRAYS=('DILE_FC','DILE_FW','LF_OVRC','ST_OVRC','RT_OVRC','LFPT','STPT','RTPT','GRAINPT','LFCT','STCT','RTCT')


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();routines={k:extract(text,k) for k in ('CARBON_CROP','CO2FLUX_CROP','GROWING_GDD')}
    body='\n'.join(routines.values());fields=set(re.findall(r'parameters%(\w+)',body,re.I))
    integers={'NROOT','ISWATER','ISBARREN','ISICE','PLTDAY','HSDAY'}
    definitions='\n'.join(('logical' if k.lower()=='urban_flag' else 'integer' if k.upper() in integers else 'real')+' :: '+k+
        ('(4)' if k.upper()=='SMCMAX' else '(8)' if k.upper() in ARRAYS else '') for k in sorted(fields))
    (build/'reference.f90').write_text('module reference\nimplicit none\ntype noahmp_parameters\n'+definitions+'\nend type\ncontains\n'+body+'\nend module\n')
    (build/'driver.f90').write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: n,i,j,pgs,nstep
real :: dt,tv,t2m,julian,psn,btran,foln,lai,sai,lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,gdd
real :: gpp,npp,nee,autors,heters,totsc,totlb,zsoil(4),dz(-2:4),stc(-2:4),smc(4)
p%ISWATER=17;p%ISBARREN=16;p%ISICE=15;p%urban_flag=.false.;p%NROOT=3;p%SMCMAX=.45
zsoil=[-.1,-.4,-1.,-2.];dz=[0.,0.,0.,.1,.3,.6,1.];smc=.2
read(*,*)n,nstep
do i=1,n
read(*,*)dt,tv,t2m,julian,psn,btran,foln,lai,sai,lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,gdd
stc=tv
read(*,*)'''+','.join('p%'+k for k in SCALARS+ARRAYS)+'''
do j=1,nstep
call CARBON_CROP(p,3,4,12,dt,zsoil,julian,dz,stc,smc,tv,psn,foln,btran,200.,t2m,lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,lai,sai,gdd,gpp,npp,nee,autors,heters,totsc,totlb,pgs)
write(*,'(18(ES25.16E3,1X))')lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,gdd,real(pgs),lai,sai,gpp,npp,nee,autors,heters,totsc,totlb
enddo
enddo
end program
''')
    fc=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-ffree-form','-ffree-line-length-none','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    subprocess.run([fc,*flags,'reference.f90','driver.f90','-o','oracle.exe'],cwd=build,check=True,timeout=120)
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    wrf_root=next(p for p in source.parents if (p/'run/MPTABLE.TBL').exists());table=wrf_root/'run/MPTABLE.TBL'
    cp=_parse_mptable_block(table.read_text(),'noahmp_crop_parameters');veg=_parse_mptable_block(table.read_text(),'noahmp_modis_parameters')
    cases=[]
    for crop in range(1,6):
        parameters={k:(veg['MRP'][11] if k=='MRP' else cp[k][crop-1]) for k in SCALARS}
        parameters.update({k:[cp[k+'_S'+str(s)][crop-1] for s in range(1,9)] for k in ARRAYS})
        for stage in range(1,9):
            gdd=0. if stage<3 else parameters['GDDS'+str(stage-2)]+1. if stage<8 else 2000.
            julian=parameters['PLTDAY']-1. if stage==1 else parameters['HSDAY'] if stage==8 else parameters['PLTDAY']+1.
            tv=260. if stage==1 else 310. if stage==7 else 300.
            cases.append(dict(crop=crop,stage=stage,dt=1800. if stage>=3 else 18.,tv=tv,t2m=tv,julian=julian,
                psn=0. if stage in (1,7,8) else 15.,btran=.01 if stage==6 else .8,foln=1.,lai=2.,sai=.2,
                state=[100.,500.,66.666664,500.,1000.,1000.,50.,gdd],parameters=parameters))
    nstep=4
    def run(rows):
        lines=[f'{len(rows)} {nstep}']
        for c in rows:
            lines += [' '.join(map(str,[c[k] for k in ('dt','tv','t2m','julian','psn','btran','foln','lai','sai')]+c['state'])),
                ' '.join(str(v) for k in SCALARS+ARRAYS for v in (c['parameters'][k] if k in ARRAYS else [int(c['parameters'][k]) if k in ('PLTDAY','HSDAY') else c['parameters'][k]]))]
        r=subprocess.run([str((build/'oracle.exe').resolve())],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=60)
        out=np.fromstring(r.stdout,sep=' ',dtype=np.float32).reshape(len(rows),nstep,18);assert np.isfinite(out).all();return out
    expected=run(cases);variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ('dt','tv','t2m','julian','psn','btran','foln','lai','sai','state'):
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
    target.write_text(json.dumps(dict(source=str(source),source_commit='f52c197ed39d12e087d02c50f412d90d418f6186',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,cases=cases,scalar_keys=SCALARS,array_keys=ARRAYS,steps=nstep,ulp_variants=len(variants),
        nominal_tolerance=dict(atol=1.e-6,rtol=3.e-6),scope='40 constructed columns, actual5crop types/all8growthstages/cold/hot/harvest/waterstress;4steps;source3routines'),indent=1)+'\n')


if __name__=='__main__':main()
