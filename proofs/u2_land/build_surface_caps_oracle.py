"""Untouched WRF VEGE_FLUX/BARE_FLUX snow-cap blocks at active savepoints."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np


def main():
    source,build,out=map(Path,sys.argv[1:4]);build.mkdir(parents=True,exist_ok=True)
    text=source.read_text();blocks={}
    for name in ('VEGE_FLUX','BARE_FLUX'):
        start=text.index('  SUBROUTINE '+name)
        start=text.index('     IF(OPT_STC == 1 .OR. OPT_STC == 3) THEN',start)
        end=text.index('! wind stresses',start)
        blocks[name]=text[start:end]
    keys='snowh temp fsno cir emg emv lwdn tv csh tah cev estg rhsur eah sag pah ir sh ev gh sfctmp'.split()
    code='''program oracle
implicit none
real,parameter::TFRZ=273.16,SB=5.67e-8
integer::i,n,kind,opt_stc
real::'''+','.join(keys)+''',tg,tgb,irg,irb,shg,shb,evg,evb,ghb,pahg,pahb,eair
read(*,*)n
do i=1,n
read(*,*)kind,opt_stc,'''+','.join(keys)+'''
tg=temp;tgb=temp;irg=ir;irb=ir;shg=sh;shb=sh;evg=ev;evb=ev;ghb=gh;pahg=pah;pahb=pah;eair=eah
if(kind==1)then
'''+blocks['VEGE_FLUX']+'''write(*,'(5(ES26.17,1X))')tg,irg,shg,evg,gh
else
'''+blocks['BARE_FLUX']+'''write(*,'(5(ES26.17,1X))')tgb,irb,shb,evb,ghb
endif
enddo
end program
'''
    flags=['-O0','-ffree-line-length-none','-ffp-contract=off','-fno-fast-math','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    (build/'reference.f90').write_text(code)
    compiler=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    assert Path(compiler).is_file(),'A Fortran compiler is required'
    subprocess.run([compiler,*flags,str(build/'reference.f90'),'-o',str(build/'oracle')],check=True,timeout=30)
    cases=[]
    for kind in (1,2):
        for stc in (1,2,3):
            for i in range(8):
                values=[(.0,.05,.3,.8)[i%4],(272.,273.16,np.nextafter(np.float32(273.16),np.float32(np.inf)),280.)[i//2],
                    (0.,.25,.7,1.)[i%4],4.8e-8,.96,.95,310.,285.,12.,275.,.0008,700.,.8,450.,120.,3.,-20.,1.5,2.,-1.,274.]
                cases.append(dict(kind=kind,opt_stc=stc,inputs=list(map(float,np.asarray(values,np.float32)))))
    rows=[];owners=[]
    for index,c in enumerate(cases):
        x=np.asarray(c['inputs'],np.float32);rows.append((c,x));owners.append(index)
        for k in range(len(keys)):
            for sign in (-np.inf,np.inf):
                variant=x.copy();variant[k]=np.nextafter(variant[k],np.float32(sign));rows.append((c,variant));owners.append(index)
    data=str(len(rows))+'\n'+'\n'.join(f"{c['kind']} {c['opt_stc']} "+' '.join(format(float(v),'.17g') for v in x) for c,x in rows)+'\n'
    result=subprocess.run([str(build/'oracle')],input=data,text=True,capture_output=True,check=True,timeout=60)
    outputs=np.asarray([[float(v) for v in line.split()] for line in result.stdout.splitlines()],np.float32)
    assert outputs.shape==(len(rows),5) and np.isfinite(outputs).all()
    for index,c in enumerate(cases):
        chosen=outputs[np.asarray(owners)==index];c.update(expected=chosen[0].tolist(),low=chosen.min(axis=0).tolist(),high=chosen.max(axis=0).tolist())
    active=int(sum(c['opt_stc'] in (1,3) and c['inputs'][0]>float(np.float32(.05)) and c['inputs'][1]>np.float32(273.16) for c in cases))
    assert active>=8
    out.write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        body_sha256={key:hashlib.sha256(value.encode()).hexdigest() for key,value in blocks.items()},flags=flags,compiler=compiler,
        keys=keys,cases=cases,active_caps=active,ulp_variants=len(rows)-len(cases),
        nominal_tolerance=dict(rtol=3.e-6,atol=2.e-6),variables_out=['TEMP','IR','SH','EV','GH'],
        units=dict(TEMP='K',IR='W m-2',SH='W m-2',EV='W m-2',GH='W m-2'),
        scope='48 constructed final-Newton savepoints; original untouched VEGE/BARE caps forSTC1/2/3, snow/temperature thresholds and fractional cover; no full-forecast claim'),indent=1)+'\n')


if __name__=='__main__':main()
