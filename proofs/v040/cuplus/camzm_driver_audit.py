"""Complete original CAM-ZM caller oracle with WRF CAM_INIT and pairings.

Fresh complete cu_camzm/driver sources plus the exact original cldwat_fice
dependency are compiled privately. Other original CAM support objects are
copied into the lane build and hashed with their sources. No numerical stubs.
The surface/PBL perturbation path exercises CAM-UW(9) and MYJ(2)+surface(2).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from generate import extract
from kfcup_driver_audit import scan_output


ROOT=Path(__file__).resolve().parents[3]
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')


def driver_source():
    original=(WRF/'phys/module_cu_camzm_driver.F').read_text()
    kf=(ROOT/'proofs/p0_4/oracle/kf_oracle_driver.f90').read_text()
    cam=(ROOT/'proofs/v034/camuw_oracle/camuw_oracle_driver.f90').read_text()
    imports=cam[cam.index('  USE shr_kind_mod'):cam.index('  IMPLICIT NONE')]
    imports=''.join(line for line in imports.splitlines(keepends=True) if 'USE module_bl_camuwpbl_driver' not in line)
    constants=kf[kf.index('  ! WRF model constants'):kf.index('  ! Single-column domain')]
    decl=original[original.index('  logical, intent(in)    ::    is_CAMMGMP_used'):original.index('! Local variables...')]
    a=decl.index('  INTEGER, INTENT(IN   )');b=decl.index('  REAL, DIMENSION(',a)
    decl=decl[:a]+'  integer :: bl_pbl_physics,itimestep,sf_sfclay_physics,stepcu\n\n'+decl[b:]
    decl=re.sub(r',\s*INTENT\s*\([^)]*\)','',decl,flags=re.I)
    decl=re.sub(r',\s*OPTIONAL','',decl,flags=re.I)
    clean='\n'.join(line.split('!')[0] for line in decl.splitlines())
    statements=re.sub(r'&\s*\n\s*','',clean)
    statements=re.sub(r'\n\s*&','',statements)
    names=[];dimensions={}
    for line in statements.splitlines():
        if '::' not in line:continue
        left,right=line.split('::',1)
        for name in right.split(','):
            name=name.strip()
            if not name:continue
            assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',name),(left,name)
            names.append(name)
            if 'DIMENSION' in left.upper():
                dimensions[name]=3 if 'kms:kme' in left else 2
    signature=original[original.index('SUBROUTINE camzm_driver('):original.index('! This routine is based')]
    signature=''.join(line.split('!')[0] for line in signature.splitlines() if not line.lstrip().startswith('#'))
    signature=signature[signature.index('(')+1:signature.rindex(')')].replace('&','').replace('\n','')
    arguments=[name.strip() for name in signature.split(',') if name.strip()]
    assert set(arguments)<=set(names)|{'ids','ide','jds','jde','kds','kde','ims','ime','jms','jme','kms','kme','its','ite','jts','jte','kts','kte'}
    bounds='''  integer,parameter::KX=40
  integer,parameter::ids=1,ide=2,jds=1,jde=2,kds=1,kde=KX+1
  integer,parameter::ims=1,ime=1,jms=1,jme=1,kms=1,kme=KX+1
  integer,parameter::its=1,ite=1,jts=1,jte=1,kts=1,kte=KX
'''
    locals_='''  real::rho(1,KX+1,1),zlev(KX),w0preset,background_wind
  integer::case_id,k,mm,n
  character(len=32)::arg
'''
    setup=''.join(f'  {name}=0\n' for name in names if name.lower() not in ('is_cammgmp_used','adapt_step_flag'))
    setup+='''  is_CAMMGMP_used=.false.;adapt_step_flag=.false.
  case_id=1;bl_pbl_physics=9;sf_sfclay_physics=1;background_wind=2.
  if(command_argument_count()>=1)then
    call get_command_argument(1,arg);read(arg,*)case_id
  endif
  if(command_argument_count()>=2)then
    call get_command_argument(2,arg);read(arg,*)bl_pbl_physics
  endif
  if(bl_pbl_physics==2)sf_sfclay_physics=2
  dt=54.;dx=9000.;cudt=0.;stepcu=5;itimestep=1;curr_secs=0.;cudtacttime=0.
  call build_sounding(case_id,t_phy,qv,p,dz8w,rho,u_phy,v_phy,zlev,w0preset)
  u_phy(:,1:KX,:)=u_phy(:,1:KX,:)+background_wind
  do k=1,KX
    pi_phy(1,k,1)=(p(1,k,1)/100000.)**ROVCP
    th(1,k,1)=t_phy(1,k,1)/pi_phy(1,k,1)
    z(1,k,1)=zlev(k)
  enddo
  ht=0.;xland=1.;mavail=1.;pblh=1500.;kpbl=6
  hfx=150.;qfx=1.e-4;ust=.3;tke_pbl=1.;tpert_camuwpbl=.3;tsk=t_phy(1,1,1)
  psfc=100000.
  if(case_id==3)psfc=100800.
  p8w(1,1,1)=psfc(1,1);z_at_w(1,1,1)=0.
  do k=2,KX
    p8w(1,k,1)=sqrt(p(1,k-1,1)*p(1,k,1))
  enddo
  p8w(1,KX+1,1)=p(1,KX,1)*p(1,KX,1)/p8w(1,KX,1)
  do k=1,KX
    z_at_w(1,k+1,1)=z_at_w(1,k,1)+dz8w(1,k,1)
  enddo
'''
    initialization=cam[cam.index('  pcnst_runtime = 5'):cam.index('  restart = .FALSE.')]
    initialization+='''  call zm_conv_init(dt,dx,rucuten,rvcuten,rthcuten,rqvcuten,rqccuten,rqicuten,rqcncuten,rqincuten, &
      2,3,4,5,1,.false.,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
'''
    # All caller data used by the oracle are dumped before and after the call.
    inputs=('th','t_phy','qv','qc','qi','p','p8w','pi_phy','u_phy','v_phy','z','z_at_w','dz8w','tke_pbl','cldfra')
    dump="  write(*,'(A,I0)')'CASE=',case_id\n"
    for name in inputs:
        count=',KX+1' if name in ('p8w','z_at_w') else ''
        dump+=f"  call dump_profile('IN_{name.upper()}',{name}{count})\n"
    call='  call camzm_driver( &\n'+', &\n'.join('    '+name+'='+name for name in arguments)+')\n'
    primary=('rucuten','rvcuten','rthcuten','rqvcuten','rqccuten','rqicuten','rqcncuten','rqincuten',
             'dlf_out','mu_out','md_out','zmdt','zmdq','zmdice','zmdliq','cmfmc','zmflxprc','zmflxsnw')
    scalars=('raincv','pratec','cape_out','cutop','cubot','pconvb','pconvt','rliq_out')
    for name in primary:
        count=',KX+1' if name=='cmfmc' else ''
        call+=f"  call dump_profile('{name.upper()}',{name}{count})\n"
    for name in scalars:call+=f"  write(*,'(A,ES23.15)')'{name.upper()}=',{name}(1,1)\n"
    contains='''contains
  subroutine dump_profile(name,arr,count)
    character(len=*),intent(in)::name
    real,intent(in)::arr(ims:ime,kms:kme,jms:jme)
    integer::kk,last
    integer,optional::count
    last=KX
    if(present(count))last=count
    do kk=1,last
      write(*,'(A,A,I0,A,ES23.15)')name,'[',kk,']=',arr(1,kk,1)
    enddo
  end subroutine
'''
    builder=kf[kf.index('  SUBROUTINE build_sounding'):kf.index('END PROGRAM kf_oracle')]
    return ('program camzm_probe\n  use module_cu_camzm_driver,only:camzm_driver,zm_conv_init\n'+imports+
            '  implicit none\n'+constants+bounds+decl+locals_+setup+initialization+dump+call+contains+builder+'end program\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-bank',type=Path)
    p.add_argument('--sequence',action='store_true')
    p.add_argument('--rotate-wind',type=float,default=0.)
    p.add_argument('--initial-real',choices=('zero','snan'),default='zero',
                   help='standing oracle policy; snan sensitivity, FP traps off')
    p.add_argument('--trace-parcel',action='store_true',help='PRINT-only dilute parcel input/output capture')
    p.add_argument('--trace-core',action='store_true',help='PRINT-only complete zm_convr input/output capture')
    p.add_argument('--optimization',choices=('O0','O2'),default='O2')
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    build=args.build_dir.resolve();build.mkdir(parents=True,exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp')
    support=('module_cam_constituents','module_cam_support','module_cam_wv_saturation',
             'module_cam_physconst','module_cam_shr_const_mod','module_cam_shr_kind_mod',
             'module_cam_esinti','module_cam_gffgch','module_cam_upper_bc')
    hashes={}
    for name in support:
        for suffix in ('.F','.o'):
            original=WRF/'phys'/(name+suffix)
            shutil.copyfile(original,build/original.name)
            hashes[str(original.relative_to(WRF))]=hashlib.sha256(original.read_bytes()).hexdigest()
    for original in (WRF/'phys/module_cu_camzm.F',WRF/'phys/module_cu_camzm_driver.F',
                     WRF/'share/module_model_constants.F',WRF/'frame/module_state_description.o'):
        shutil.copyfile(original,build/original.name)
        hashes[str(original.relative_to(WRF))]=hashlib.sha256(original.read_bytes()).hexdigest()
    source,bounds=extract((WRF/'phys/module_cam_cldwat.F').read_text(),'cldwat_fice')
    if args.trace_parcel:
        path=build/'module_cu_camzm.F';text=path.read_text()
        start=text.index('subroutine buoyan_dilute(')
        end=text.index('end subroutine buoyan_dilute',start)
        ret=text.rfind('   return',start,end)
        vectors=('q','t','p','z','pf','tp','qstp')
        capture='\n'.join(f"do k=1,pver{'p' if name=='pf' else ''}\n write(0,'(A,I0,A,ES26.17E3)') 'ZM_PARCEL_{name}[',k,']=',{name}(1,k)\nenddo"
                           for name in vectors)+'\n'
        capture+='\n'.join(f"write(0,'(A,ES26.17E3)') 'ZM_PARCEL_{name}=',{name}(1)" for name in ('tl','cape','pblt','tpert'))+'\n'
        capture+='\n'.join(f"write(0,'(A,I0)') 'ZM_PARCEL_{name}=',{name}(1)" for name in ('lcl','lel','lon','mx'))+'\n'
        capture+="write(0,'(A,I0)') 'ZM_PARCEL_msg=',msg\n"
        text=text[:ret]+capture+text[ret:];path.write_text(text)
    if args.trace_core:
        path=build/'module_cu_camzm.F';text=path.read_text()
        start=text.index('subroutine zm_convr(');end=text.index('end subroutine zm_convr',start)
        def vectors(prefix,names):
            return '\n'.join(f"do k=1,pver{'p' if name in ('paph','zi','mcon','pflx') else ''}\n write(0,'(A,I0,A,ES26.17E3)') '{prefix}{name}[',k,']=',{name}(1,k)\nenddo" for name in names)+'\n'
        inputs=vectors('ZM_CORE_IN_',('t','pap','paph','dpp','zm','zi'))
        inputs+="do k=1,pver\n write(0,'(A,I0,A,ES26.17E3)') 'ZM_CORE_IN_q[',k,']=',qh(1,k,1)\nenddo\n"
        inputs+='\n'.join(f"write(0,'(A,ES26.17E3)') 'ZM_CORE_CTRL_{name}=',{name}(1)" for name in ('geos','pblh','tpert','landfrac'))+'\n'
        inputs+="write(0,'(A,ES26.17E3)') 'ZM_CORE_CTRL_dt=',delt\nwrite(0,'(A,ES26.17E3)') 'ZM_CORE_CTRL_tau=',tau\n"
        inputs+="write(0,'(A,I0)') 'ZM_CORE_CTRL_limcnv=',limcnv\n"
        outputs=vectors('ZM_CORE_OUT_',('qtnd','heat','mcon','dlf','pflx','cme','rprd','zdu','ql','mu','md','du','eu','ed','dp'))
        outputs+='\n'.join(f"write(0,'(A,ES26.17E3)') 'ZM_CORE_OUT_{name}=',{name}(1)" for name in ('cape','prec','rliq','jctop','jcbot','dsubcld'))+'\n'
        outputs+='\n'.join(f"write(0,'(A,I0)') 'ZM_CORE_OUT_{name}=',{name}(1)" for name in ('jt','maxg'))+'\n'
        outputs+="write(0,'(A,I0)') 'ZM_CORE_OUT_lengath=',lengath\n"
        block=text[start:end]
        block=block.replace('   msg = limcnv - 1',inputs+'   msg = limcnv - 1',1)
        assert block.count('   if (lengath.eq.0) return')==1
        block=block.replace('   if (lengath.eq.0) return','   if (lengath.eq.0) then\n'+outputs+'   return\n   endif')
        ret=block.rfind('   return')
        block=block[:ret]+outputs+block[ret:]
        text=text[:start]+block+text[end:];path.write_text(text)
    (build/'cldwat_dependency.f90').write_text('''module cldwat
use shr_kind_mod,only:r8=>shr_kind_r8
use physconst,only:tmelt
use module_cam_support,only:pcols,pver
implicit none
real(r8),private::tmax_fice,tmin_fice,tmax_fsnow,tmin_fsnow
contains
'''+source+'\nend module\n')
    shutil.copyfile(ROOT/'proofs/v034/camuw_oracle/wrf_runtime_stubs.f90',build/'logger.f90')
    with (build/'logger.f90').open('a') as logger:
        logger.write('''
subroutine wrf_error_fatal(message)
  implicit none
  character(len=*),intent(in)::message
  call wrf_error_fatal3('camzm_oracle',0,message)
end subroutine
''')
    bank=None;driver=driver_source()
    if args.profile_bank:
        from real_column_bank import prepare,bind
        bank=prepare(args.profile_bank,build);driver=bind(driver,bank,'camzm')
    sequence=None
    if args.sequence:
        from sequence_oracle import bind as sequence_bind
        driver,sequence=sequence_bind(driver,'camzm')
    from wind_rotation import bind as rotate
    driver=rotate(driver,'camzm',args.rotate_wind)
    (build/'driver.f90').write_text(driver)
    compiler=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-'+args.optimization,'-g','-fbacktrace','-ffree-form','-ffree-line-length-none','-cpp','-DWRF_PORT','-DEM_CORE=1',
           '-finit-real='+args.initial_real,'-finit-integer=0','-finit-logical=false',
           '-I'+str(WRF/'phys'),'-I'+str(WRF/'share'),'-I'+str(WRF/'frame')]
    objects=[name+'.o' for name in support]+['module_state_description.o']
    sources=['logger.f90','module_model_constants.F','cldwat_dependency.f90',
             'module_cu_camzm.F','module_cu_camzm_driver.F','driver.f90']
    record=dict(source_object_sha256=hashes,cldwat_fice=dict(lines=bounds,sha256=hashlib.sha256(source.encode()).hexdigest()),
                driver_sha256=hashlib.sha256((build/'driver.f90').read_bytes()).hexdigest(),flags=flags,records=[],stage='COMPILING',profile_bank=bank,sequence=sequence,wind_rotation_degrees=args.rotate_wind)
    record['instrumentation']=('PRINT-only zm_convr' if args.trace_core else
                              ('PRINT-only dilute parcel' if args.trace_parcel else 'NONE'))
    record['compiled_camzm_sha256']=hashlib.sha256((build/'module_cu_camzm.F').read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic()
    subprocess.run([compiler,*flags,*sources,*objects,'-lmvec','-lm','-o','oracle'],cwd=build,check=True,timeout=900)
    record['compile_s']=time.monotonic()-started
    for case in (bank['cases'] if bank else (1,2,3,4)):
        for pbl in (9,2):
            result=subprocess.run([str(build/'oracle'),str(case),str(pbl)],cwd=build,capture_output=True,text=True,timeout=120)
            stdout=build/f'case{case}-pbl{pbl}.stdout';stderr=stdout.with_suffix('.stderr')
            stdout.write_text(result.stdout);stderr.write_text(result.stderr)
            record['records'].append(dict(case=case,pbl=pbl,surface=2 if pbl==2 else 1,
                                          returncode=result.returncode,stdout=str(stdout),stderr=str(stderr),
                                          output=scan_output(result.stdout) if result.returncode==0 else None))
            record['stage']='RUNNING';args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    record['stage']='COMPLETE';args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,allow_nan=False))


if __name__=='__main__':
    main()
