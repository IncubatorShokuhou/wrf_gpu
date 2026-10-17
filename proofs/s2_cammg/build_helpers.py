"""Original CAM-MG cloud and gamma helpers on independent threshold operands."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from build_reference import WRF,FC


def body(text,name,kind='subroutine'):
    import re
    start=re.search(r'^\s*'+kind+r'\s+'+name+r'\s*\(',text,re.M|re.I).start()
    end=re.search(r'^\s*end\s+'+kind+r'\s+'+name+r'\b',text[start:],re.M|re.I).end()+start
    return text[start:end]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('build',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();args.build.mkdir(parents=True,exist_ok=True)
    driver=(WRF/'phys/module_mp_cammgmp_driver.F').read_text()
    core=(WRF/'phys/module_cam_mp_cldwat2m_micro.F').read_text()
    original=[body(driver,'cal_cldfra_mp_1d'),body(core,'gamma','function')]
    helper='module original_helpers\nuse shr_kind_mod,only:r8=>shr_kind_r8\ncontains\n'+'\n'.join(original)+'\nend module\n'
    (args.build/'helpers.f90').write_text(helper)
    inputs=[]
    from math import exp
    # Values straddle each liquid-fraction branch, pressure threshold and phase
    # boundary. Saturation values come from ORIGINAL WRF polysvp in the wrapper.
    for t in [215.,240.,263.15,273.149999999,273.15,273.150000001,285.,305.]:
        for p in [39999.,40000.,50000.,69999.,70000.,100000.]:
            for land,snow in [(1.,0.),(1.,.01),(2.,0.)]:
                for rh in [0.,.5,.79,.80,.85,.90,.96,.98,1.,1.05]:
                    inputs.append([t,p,land,snow,rh,1.e-4 if t<273.15 else 0.])
    inputs=np.asarray(inputs,dtype=np.float64)
    with (args.build/'inputs.txt').open('w') as f:
        f.write(str(len(inputs))+'\n')
        np.savetxt(f,inputs,fmt='%.17e')
    program='''program test
use original_helpers
use wv_saturation,only:polysvp,epsqs,vqsatd_water
use module_cam_esinti,only:esinti
use physconst,only:epsilo,latvap,latice,rh2o,cpair,tmelt
implicit none
integer::n,i
real(r8)::t,p,xland,snowh,rh,qi,qqv,qqc,cldl,cldi,esl,esi,qsat
real(r8)::es(1),qs(1),gam(1),ga
open(10,file='inputs.txt',status='old');read(10,*)n
call esinti(epsilo,latvap,latice,rh2o,cpair,tmelt)
do i=1,n
read(10,*)t,p,xland,snowh,rh,qi
esl=polysvp(t,0);esi=polysvp(t,1)
qsat=epsqs*esl/(p-(1._r8-epsqs)*esl)
qqv=rh*qsat;qqc=0._r8
cldl=0._r8;cldi=0._r8
call cal_cldfra_mp_1d(cldl,cldi,qqv,qqc,qi,t,p,xland,snowh)
call vqsatd_water([t],[p],es,qs,gam,1)
ga=gamma(1._r8+mod(i,140)/10._r8)
write(*,'(*(ES26.17E3,1X))') t,p,xland,snowh,qqv,qqc,qi,cldl,cldi,esl,esi,es,qs,gam,ga
enddo
end program
'''
    (args.build/'driver.f90').write_text(program)
    flags=['-O2','-ffp-contract=off','-ffree-form','-ffree-line-length-none',f'-I{WRF}/phys',f'-I{WRF}/share',f'-I{WRF}/frame']
    (args.build/'stubs.f90').write_bytes(Path('proofs/v034/camuw_oracle/wrf_runtime_stubs.f90').read_bytes())
    objs=[]
    for name in ['helpers','driver','stubs']:
        obj=args.build/f'{name}.o'
        subprocess.run([str(FC),*flags,'-c',str(args.build/f'{name}.f90'),'-o',str(obj)],cwd=args.build,check=True)
        objs.append(obj)
    deps=['module_cam_wv_saturation','module_cam_physconst','module_cam_shr_const_mod','module_cam_shr_kind_mod','module_cam_esinti','module_cam_support','module_cam_gffgch']
    subprocess.run([str(FC),*map(str,objs),*[str(WRF/'phys'/f'{n}.o') for n in deps],'-lm','-o',str(args.build/'test.exe')],check=True)
    raw=subprocess.check_output([str(args.build/'test.exe')],cwd=args.build,text=True)
    (args.build/'reference.txt').write_text(raw)
    values=np.loadtxt([s for s in raw.splitlines() if len(s.split())==15]);assert values.shape==(len(inputs),15)
    assert np.all(np.isfinite(values))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,inputs=values[:,:7],cloud=values[:,7:9],saturation=values[:,9:14],
                        gamma_x=1.+np.arange(1,len(inputs)+1)%140/10.,gamma=values[:,14])
    manifest={'source_sha256':[hashlib.sha256(b.encode()).hexdigest() for b in original],
              'rows':len(inputs),'dtype':'float64 (original CAM r8)','finite':True,
              'source':'WRF4.7.1 original cal_cldfra_mp_1d/gamma bodies; native wv_saturation polysvp/vqsatd_water object',
              'flags':flags,'asset_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'literal_gate':'BYTE vs original no-FMA; compiled gate must use independent controls, not retuned tolerance'}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
