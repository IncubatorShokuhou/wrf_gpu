"""Build/run exact original MSKF_CPS to establish a full caller oracle.

This diagnostics driver adapts the existing original-KF sounding builder,
not its numerics, uses the entire unmodified MSKF source, default CMT ON,
and supplies the W staggered top level missing from the old KF harness.
No full WRF executable is run (no shared E104 dump writes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT=Path(__file__).resolve().parents[3]


def replace_once(text, old, new):
    if text.count(old)!=1:
        raise ValueError(f"driver anchor missing/ambiguous: {old[:70]}")
    return text.replace(old,new)


def driver_source():
    text=(ROOT/'proofs/p0_4/oracle/kf_oracle_driver.f90').read_text()
    text=replace_once(text,'USE module_cu_kfeta, ONLY : kf_eta_cps, kf_eta_init',
                      'USE module_cu_mskf, ONLY : mskf_cps, mskf_init')
    text=text.replace('kf_eta_init','mskf_init').replace('KF_ETA_CPS','MSKF_CPS')
    # Proper top interface allocation: CPS averages W(k)+W(k+1) even for
    # the uppermost mass level. Horizontal dimensions include caller halos.
    text=replace_once(text,'ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX',
                      'ims=1, ime=5, jms=1, jme=5, kms=1, kme=KX+1')
    text=replace_once(text,'ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1',
                      'ids=1, ide=6, jds=1, jde=6, kds=1, kde=KX+1')
    text=replace_once(text,'its=1, ite=1, jts=1, jte=1, kts=1, kte=KX',
                      'its=2, ite=4, jts=2, jte=4, kts=1, kte=KX')
    text=replace_once(text,'INTEGER :: k, case_id','INTEGER :: k, case_id, ii, jj')
    anchor='REAL, DIMENSION(KX) :: zlev'
    text=replace_once(text,anchor,anchor+'''
  REAL :: BACKGROUND_WIND
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: QR_KF,QS_KF,NC_KF,NI_KF,NR_KF,NS_KF,CCN_KF
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: EFCS,EFIS,EFSS,RUCUTEN,RVCUTEN
  REAL, DIMENSION(ims:ime,jms:jme) :: ZOL,HFX,UST,PBLH,XLAND,AINC_FRAC
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme,10) :: AEROCU
''')
    text=replace_once(text,'  ! --- model timestep / grid', '''
  BACKGROUND_WIND=0.
  IF(COMMAND_ARGUMENT_COUNT() >= 2) THEN
    CALL GET_COMMAND_ARGUMENT(2,arg)
    READ(arg,*) BACKGROUND_WIND
  ENDIF
  ! --- model timestep / grid''')
    text=replace_once(text,'its,ite,jts,jte,kts,kte)\n\n  ! --- build',
                      'its,ite,jts,jte,kts,kte,RUCUTEN,RVCUTEN)\n\n  ! --- build')
    # Initialize ONLY caller-owned input/output fields. Original procedure
    # locals are never initialized by this driver; the poisoned build exposes
    # any dependence on unspecified local storage.
    anchor='  CALL MSKF_CPS( &'
    setup='''
  DO jj=jms,jme
    DO ii=ims,ime
      DO k=kts,kte
        T(ii,k,jj)=T(1,k,1); QV(ii,k,jj)=QV(1,k,1)
        PCPS(ii,k,jj)=PCPS(1,k,1); RHO(ii,k,jj)=RHO(1,k,1)
        DZ8W(ii,k,jj)=DZ8W(1,k,1); U(ii,k,jj)=U(1,k,1); V(ii,k,jj)=V(1,k,1)
        PII(ii,k,jj)=PII(1,k,1); TH(ii,k,jj)=TH(1,k,1)
        W(ii,k,jj)=W(1,k,1); W0AVG(ii,k,jj)=W0AVG(1,k,1)
        RQVFTEN(ii,k,jj)=0.
      ENDDO
      W(ii,KX+1,jj)=0.
    ENDDO
  ENDDO
  RTHCUTEN=0.; RQVCUTEN=0.; RQCCUTEN=0.; RQRCUTEN=0.; RQICUTEN=0.; RQSCUTEN=0.
  RUCUTEN=0.;RVCUTEN=0.;QR_KF=0.;QS_KF=0.;NC_KF=0.;NI_KF=0.;NR_KF=0.;NS_KF=0.;CCN_KF=0.
  EFCS=2.51;EFIS=5.01;EFSS=10.01;AINC_FRAC=0.;AEROCU=0.
  ZOL=-.1;HFX=150.;UST=.3;PBLH=1000.;XLAND=1.;NCA=-100.
'''
    # U has the common W allocation (kme=KX+1), but only its MASS levels
    # exist in this sounding. Never touch the unused extra U level.
    setup += '  U(:,kts:kte,:)=U(:,kts:kte,:)+BACKGROUND_WIND\n'
    text=replace_once(text,anchor,setup+anchor)
    text=replace_once(text,'QV, SHALL, &','QV, &')
    text=replace_once(text,'QC_KF,QI_KF, &','QC_KF,QI_KF,QR_KF,QS_KF,NC_KF,NI_KF,NR_KF,NS_KF,CCN_KF,AINC_FRAC, &')
    text=replace_once(text,'TIMEC_KF,KF_EDRATES )','TIMEC_KF,KF_EDRATES,ZOL,HFX,UST,PBLH,AEROCU,10,1.0,0,EFCS,EFIS,EFSS,RUCUTEN,RVCUTEN,XLAND )')
    # Dump the center tile point actually run by CPS (not the untouched halo).
    text=text.replace("arr(1,kk,1)","arr(3,kk,3)")
    for name in ('RAINCV','PRATEC','NCA','CUTOP','CUBOT','TIMEC_KF'):
        # Keep the caller NCA seed above: all-grid initialization supersedes
        # it. Only printed references are adapted by line selection.
        lines=text.splitlines(keepends=True)
        text=''.join(l.replace(name+'(1,1)',name+'(3,3)') if l.lstrip().upper().startswith('WRITE') else l for l in lines)
    text=replace_once(text,"CALL dump_col('RQSCUTEN', RQSCUTEN)","""CALL dump_col('RQSCUTEN', RQSCUTEN)
  CALL dump_col('RUCUTEN', RUCUTEN)
  CALL dump_col('RVCUTEN', RVCUTEN)
  CALL dump_col('CLDFRA_DP', CLDFRA_DP_KF)
  CALL dump_col('CLDFRA_SH', CLDFRA_SH_KF)
  CALL dump_col('QC_KF', QC_KF)
  CALL dump_col('QI_KF', QI_KF)
  CALL dump_col('PI', PII)""")
    # SHALL is not an MSKF argument: report its real cloud diagnostics instead.
    text=''.join(l for l in text.splitlines(keepends=True) if "'SHALL='" not in l)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-bank',type=Path)
    parser.add_argument('--sequence',action='store_true')
    parser.add_argument('--rotate-wind',type=float,default=0.)
    parser.add_argument('--optimization',choices=('O0','O2'),default='O0')
    parser.add_argument('--background-wind',type=float,action='append')
    parser.add_argument('--modes',nargs='+',choices=('plain','checked','poisoned','poisoned_nobounds'),
                        default=['plain','checked','poisoned'])
    parser.add_argument('--trace-source',action='store_true',help='print-only NINT and raw DER-bit diagnostics')
    parser.add_argument('--trace-cmt',action='store_true',help='PRINT-only original DOUBLE CMT input/output capture')
    parser.add_argument('--compile-timeout',type=int,default=900)
    parser.add_argument('--initial-real',choices=('zero','snan'),default='zero',
                        help='standing oracle policy; snan plain mode keeps FP traps off')
    args=parser.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    if Path('/tmp/wrf_gpu2_quiet').exists():raise RuntimeError('QUIET')
    build=args.build_dir.resolve();build.mkdir(parents=True,exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp')
    wrf=Path('<USER_HOME>/src/wrf_pristine/WRF')
    sources=['module_cam_shr_kind_mod.F','module_cam_error_function.F','module_cu_mskf.F']
    hashes={}
    for name in sources:
        src=wrf/'phys'/name
        shutil.copyfile(src,build/name);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    if args.trace_source:
        path=build/'module_cu_mskf.F';text=path.read_text()
        text=replace_once(text,'         NIC=NINT(TIMEC/DT)', '''         NIC=NINT(TIMEC/DT)
         write(0,'(A,2I5,3ES25.16E3,I14)') 'MSKF_NINT',I,J,VCONV,TADVEC,TIMEC,NIC''')
        text=replace_once(text,'                 DERconvF(1,JK) = (((DER(KQ)/DXSQ)*G)/100.)/DZQ(KQ)', '''
                 if(ibits(transfer(DER(KQ),0),23,8).eq.255) then
                   write(0,'(A,5I5,3Z12.8)') 'MSKF_DER_RAW',I,J,KQ,LTOP,LFS, &
                       transfer(DER(KQ),0),transfer(DZQ(KQ),0),transfer(DXSQ,0)
                 endif
                 DERconvF(1,JK) = (((DER(KQ)/DXSQ)*G)/100.)/DZQ(KQ)''')
        path.write_text(text)
    if args.trace_cmt:
        path=build/'module_cu_mskf.F';text=path.read_text()
        vectors=('DP','ED','EU','MC','MD','MU','P','QD','QU','QHAT','SD','SU','SHAT','T','U','V','Z','ZF')
        def dump(prefix,names):
            return '\n'.join(
                f"do jj=1,ILEV\n write(0,'(A,I0,A,ES26.17E3)') '{prefix}{name}[',jj,']=',{name}(IL1G,jj)\nenddo"
                for name in names)
        scalars=('DT','GRAV','CPRES','RGAS')
        capture='\n'+dump('CMT_IN_',vectors)+'\n'
        capture+='\n'.join(f"write(0,'(A,ES26.17E3)') 'CMT_CTRL_{name}=',{name}" for name in scalars)+'\n'
        capture+="write(0,'(A,ES26.17E3)') 'CMT_CTRL_DSUBCLD=',DSUBCLD(IL1G)\n"
        capture+='\n'.join(f"write(0,'(A,I0)') 'CMT_CTRL_{name}=',{name}(IL1G)" for name in ('JB','JD','JT'))+'\n'
        text=replace_once(text,'    RC=3000._r8',capture+'    RC=3000._r8')
        tail=text.index('    END SUBROUTINE MSKF_CMT')
        ret=text.rfind('    RETURN',0,tail)
        text=text[:ret]+dump('CMT_OUT_',('DUDT','DVDT','dpdx','dpdy'))+'\n'+text[ret:]
        path.write_text(text)
    shutil.copyfile(ROOT/'proofs/p0_4/oracle/module_wrf_error.f90',build/'module_wrf_error.f90')
    # The aerosol submodule (same original file) calls the external WRF
    # fatal ABI, whereas MSKF itself USEs the module procedure. Supply both
    # logging ABIs without changing a numerical source line.
    with (build/'module_wrf_error.f90').open('a') as stream:
        stream.write('''
subroutine wrf_error_fatal(message)
 use module_wrf_error, only: module_fatal=>wrf_error_fatal
 implicit none
 character(len=*),intent(in)::message
 call module_fatal(message)
end subroutine wrf_error_fatal
''')
    winds=args.background_wind or [0.]
    bank=None
    driver=driver_source()
    if args.profile_bank:
        from real_column_bank import prepare,bind
        bank=prepare(args.profile_bank,build);driver=bind(driver,bank,'mskf');winds=[0.]
    sequence=None
    if args.sequence:
        from sequence_oracle import bind as sequence_bind
        driver,sequence=sequence_bind(driver,'mskf')
    from wind_rotation import bind as rotate
    driver=rotate(driver,'mskf',args.rotate_wind)
    (build/'driver.F90').write_text(driver)
    compiler=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    optimize=['-O0'] if args.optimization=='O0' else ['-O2','-ftree-vectorize','-funroll-loops']
    common=[compiler]+optimize+['-g','-fbacktrace','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1']
    common+=['-finit-real='+args.initial_real,'-finit-integer=0','-finit-logical=false']
    source_args=['module_wrf_error.f90']+sources+['driver.F90']
    records=[]
    compile_times={}
    def write_receipt(stage):
        receipt=dict(stage=stage,source_sha256=hashes,
                     compiled_source_sha256={n:hashlib.sha256((build/n).read_bytes()).hexdigest() for n in sources},
                     instrumentation='PRINT-ONLY CMT inputs/outputs' if args.trace_cmt else
                                     ('PRINT-ONLY NINT/DER bits' if args.trace_source else 'NONE'),
                     compiler=compiler,flags=common[1:],background_winds=winds,compile_seconds=compile_times,profile_bank=bank,sequence=sequence,wind_rotation_degrees=args.rotate_wind,
                     driver_sha256=hashlib.sha256((build/'driver.F90').read_bytes()).hexdigest(),records=records)
        args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    compiler_modes={'plain':[], 'checked':['-fcheck=all'],
                    'poisoned':['-fcheck=all','-finit-real=snan','-ffpe-trap=invalid,zero,overflow'],
                    'poisoned_nobounds':['-finit-real=snan','-ffpe-trap=invalid,zero,overflow']}
    for mode in args.modes:
        extra=compiler_modes[mode]
        write_receipt('COMPILING '+mode)
        start=time.monotonic()
        subprocess.run(common+extra+source_args+['-o',mode],cwd=build,check=True,timeout=args.compile_timeout)
        compile_times[mode]=time.monotonic()-start
        for wind in winds:
            for case in (bank['cases'] if bank else (1,2,3,4)):
                result=subprocess.run([str(build/mode),str(case),str(wind)],cwd=build,capture_output=True,text=True,timeout=30)
                stdout=build/f'{mode}-w{wind:g}-{case}.stdout';stderr=build/f'{mode}-w{wind:g}-{case}.stderr'
                stdout.write_text(result.stdout);stderr.write_text(result.stderr)
                records.append(dict(mode=mode,background_wind=wind,case=case,returncode=result.returncode,stdout=str(stdout),stderr=str(stderr)))
                write_receipt('RUNNING '+mode)
    write_receipt('COMPLETE')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
