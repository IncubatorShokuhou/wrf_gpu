"""Whole original SOILWATER with VIC/XAJ/DVIC and both complete tile models."""
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from build_crop_oracle import extract


def main():
    source,build,target=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text()
    names=('SOILWATER','SRT','SSTEP','INFIL','WDFCND1','WDFCND2','ROSR12','ZWTEQ','COMPUTE_VIC_SURFRUNOFF','COMPUTE_XAJ_SURFRUNOFF',
        'DYNAMIC_VIC','RR1','RR2','PHILIP_INFIL','GREEN_AMPT_INFIL','SMITH_PARLANGE_INFIL','TILE_DRAIN','TILE_HOOGHOUDT','TD_EQUIVALENT_DEPTH','TD_FINDZWAT')
    routines={k:extract(text,k) for k in names};body='\n'.join(routines.values())
    fields=sorted(set(k.upper() for k in re.findall(r'parameters%(\w+)',body,re.I)))
    arrays=set(k.upper() for k in re.findall(r'parameters%(\w+)\(',body,re.I))
    integers={'DRAIN_LAYER_OPT','TD_DEPTH'}
    definitions='\n'.join(('logical' if k=='URBAN_FLAG' else 'integer' if k in integers else 'real')+' :: '+k+('(4)' if k in arrays else '') for k in fields)
    (build/'reference.f90').write_text('module reference\nimplicit none\ninteger :: opt_run,opt_inf,opt_tdrn,opt_infdv\ntype noahmp_parameters\n'+definitions+'\nend type\ncontains\n'+body+'\nend module\n')
    stubs=set(k.upper() for k in re.findall(r'\bCALL\s+(\w+)',body,re.I))-set(names)
    stubtext='\n'.join('subroutine '+k+'\nstop 99\nend subroutine\n' for k in sorted(stubs))
    readfields=[k for k in fields if k!='URBAN_FLAG']
    (build/'driver.f90').write_text('''program oracle
use reference
implicit none
type(noahmp_parameters) :: p
integer :: i,n
real :: dt,qinsur,qseva,sh2o(4),smc(4),sice(4),etrani(4),zsoil(4),dz(4),zwt,smcwtd,deeprech,runsrf,qdrain,runsub,wcnd(4),fcrmax,qtl
zsoil=[-.1,-.4,-1.,-2.];dz=[.1,.3,.6,1.]
open(77,file='outputs.dat',status='replace')
read(*,*)n
do i=1,n
read(*,*)opt_run,opt_inf,opt_tdrn,opt_infdv,dt,qinsur,qseva,sh2o,smc,etrani
read(*,*)'''+','.join('p%'+k for k in readfields)+'''
p%urban_flag=.false.
sice=max(0.,smc-sh2o);zwt=2.5;smcwtd=.2;deeprech=0.;runsub=0.;qtl=0.
call SOILWATER(p,4,0,dt,zsoil,dz,qinsur,qseva,etrani,sice,1,1,1.,3000.,sh2o,smc,zwt,14,smcwtd,deeprech,runsrf,qdrain,runsub,wcnd,fcrmax,qtl)
write(77,'(13(ES25.16E3,1X))')sh2o,smc,runsrf,qdrain,runsub,qtl,zwt
enddo
close(77)
end program
'''+stubtext)
    fc=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O0','-cpp','-ffree-form','-ffree-line-length-none','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    subprocess.run([fc,*flags,'reference.f90','driver.f90','-o','oracle.exe'],cwd=build,check=True,timeout=120)
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    wrf_root=next(p for p in source.parents if (p/'run/MPTABLE.TBL').exists());table=wrf_root/'run/MPTABLE.TBL'
    tile=_parse_mptable_block(table.read_text(),'noahmp_tiledrain_parameters')
    p=dict(BEXP=[4.5,5.,4.,5.5],SMCMAX=[.45,.47,.45,.43],SMCREF=[.30,.32,.30,.28],SMCWLT=[.1,.12,.1,.08],
        DKSAT=[1.e-5,2.e-6,5.e-6,1.e-6],DWSAT=[1.e-5,1.2e-5,1.5e-5,1.e-5],PSISAT=[.35,.4,.3,.45],KDT=3.,FRZX=.15,SLOPE=.1,FSATMX=.38,TIMEAN=10.5,
        BVIC=.18,AXAJ=.013,BXAJ=.18,XXAJ=.18,BDVIC=.18,BBVIC=1.,GDVIC=.11)
    aliases={'TDSMC_FAC':'TDSMC_FAC'}
    for k in fields:
        if k in p or k in ('URBAN_FLAG','DRAIN_LAYER_OPT'):continue
        table_key='TDSMC_FAC' if k=='TDSMC_FAC' else k
        if table_key not in tile:raise ValueError('Unbound source parameter '+k)
        p[k]=int(tile[table_key][5]) if k in integers else tile[table_key][5]
    modes=[(6,0,0,4),(7,0,0,4),(8,1,0,4),(8,2,0,4),(8,3,0,4)]+[(3,0,1,k) for k in range(6)]+[(3,0,2,4)]
    cases=[]
    for run,infdv,tdrn,layer in modes:
        for inf in (1,2):
            for qin,sh,ice in ((0.,.2,0.),(1.e-6,.12,0.),(1.e-4,.3,0.),(1.e-2,.3,0.),(1.e-5,.1,.2),(1.e-3,.05,.3),(0.,.005,.4),(1.e-4,.44,0.)):
                cases.append(dict(opt_run=run,opt_inf=inf,opt_infdv=infdv,opt_tdrn=tdrn,drain_layer_opt=layer,dt=1800. if tdrn==2 else 18.,qinsur=qin,qseva=1.e-8,
                    sh2o=[sh,min(sh,.47),min(sh,.45),min(sh,.43)],smc=[min(sh+ice,v) for v in p['SMCMAX']],etrani=[1.e-8,2.e-8,1.e-8,0.]))
    def run(rows):
        lines=[str(len(rows))]
        for c in rows:
            lines+=[' '.join(map(str,[c[k] for k in ('opt_run','opt_inf','opt_tdrn','opt_infdv','dt','qinsur','qseva')]+c['sh2o']+c['smc']+c['etrani'])),
                ' '.join(str(v) for k in readfields for v in ([c['drain_layer_opt']] if k=='DRAIN_LAYER_OPT' else p[k] if k in arrays else [p[k]]))]
        subprocess.run([str((build/'oracle.exe').resolve())],cwd=build,input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True,timeout=60)
        out=np.fromstring((build/'outputs.dat').read_text(),sep=' ',dtype=np.float32).reshape(len(rows),13);assert np.isfinite(out).all();return out
    expected=run(cases)
    for tile_model in (1,2):
        assert any(row[11]>0. for c,row in zip(cases,expected) if c['opt_tdrn']==tile_model),f'Inactive tile model {tile_model}'
    variants=[];owner=[];indices=[]
    for i,c in enumerate(cases):
        indices.append(len(variants));variants.append(c.copy());owner.append(i)
        for key in ('dt','qinsur','qseva','sh2o','smc','etrani'):
            for index in (range(4) if isinstance(c[key],list) else (None,)):
                for direction in (-np.inf,np.inf):
                    v=c.copy();value=c[key][index] if index is not None else c[key]
                    changed=float(np.nextafter(np.float32(value),np.float32(direction)))
                    if index is None:v[key]=changed
                    else:v[key]=c[key].copy();v[key][index]=changed
                    variants.append(v);owner.append(i)
    results=run(variants);assert np.array_equal(results[indices],expected)
    low,high=expected.copy(),expected.copy()
    for i,row in zip(owner,results):low[i]=np.minimum(low[i],row);high[i]=np.maximum(high[i],row)
    for i,c in enumerate(cases):c.update(expected=expected[i].tolist(),low=low[i].tolist(),high=high[i].tolist())
    target.write_text(json.dumps(dict(source=str(source),source_commit='f52c197ed39d12e087d02c50f412d90d418f6186',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        routine_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in routines.items()},table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),flags=flags,parameters=p,cases=cases,ulp_variants=len(variants),nominal_tolerance=dict(atol=1.e-7,rtol=3.e-6),
        unreachable_stubs=sorted(stubs),scope='192 constructed fullSOILWATER columns;run6/7/8 infdv1/2/3 andsimpletileall6layerchoices+Hooghoudt,INF1/2,8wet/frozen/rain regimes'),indent=1)+'\n')


if __name__=='__main__':main()
