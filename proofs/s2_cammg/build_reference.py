"""Build the original non-chem WRF CAM-MG public caller in a private lane dir.

The generated wrapper mirrors CAM_INIT and passes real WRF configuration/types.
No pristine Fortran routine body is changed. Public arrays retain WRF REAL4 ABI;
CAM's internal r8 is WRF DOUBLE. Run on one lane CPU with a QUIET guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
FC = Path('<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
MODULES = '''module_mp_cammgmp_driver module_cam_mp_cldwat2m_micro
module_cam_mp_microp_aero module_cam_mp_ndrop module_cam_mp_conv_water
module_cam_mp_qneg3 module_cam_mp_modal_aero_initialize_data_phys
module_data_cam_mam_aero module_data_cam_mam_asect module_cam_error_function
module_cam_cldwat module_cam_constituents module_cam_support
module_cam_wv_saturation module_cam_physconst module_cam_shr_const_mod
module_cam_shr_kind_mod module_cam_esinti module_cam_gffgch'''.split()

STAGES = {
    'macro_in': '''dtime state1_t state_pmid state_q state1_s state1_q
        dp_icwmr sh_icwmr dp_frac sh_frac qi_mac rh_old cldo xland_pt snowh_pt'''.split(),
    'macro_out': '''state1_s state1_q rh_old cldo esl qvs ast aist alst
        qi_mac ptend_all_q ptend_all_s'''.split(),
    'aero_in': '''dtime state1_t state1_q qc qi nc ni state1_pmid state1_pdel
        ast alst_mic aist_mic cldo state1_pint state1_rpdel state1_zm
        state1_omega cflx ptend_loc_q dgnumwet dgnum kkvh tke turbtype smaw qqcw'''.split(),
    'aero_out': '''ptend_loc_q qqcw wsub wsubi naai naai_hom npccn rndst nacon'''.split(),
    'micro_out': '''qc qi nc ni cldo rate1cld tlat qvlat qcten qiten ncten niten
        effc effc_fn effi prect preci nevapr evapsnow prain prodsnow cmeice dei mu
        lambdac qsout des rflx sflx qrout reff_rain reff_snow qcsevap qisevap
        qvres cmeiout vtrmc vtrmi qcsedten qisedten prao prco mnuccco mnuccto
        msacwio psacwso bergso bergo melto homoo qcreso prcio praio qireso
        mnuccro pracso meltsdt frzrdt mnuccdo nsout nrout'''.split(),
}


def observe(text):
    def dump(stage):
        return (f"\nwrite(*,'(A)') 'STAGE {stage}'\n" +
                ''.join(f"write(*,'(A,*(1X,ES26.17E3))') '{name}',{name}\n"
                        for name in STAGES[stage]) + "write(*,'(A)') 'END_STAGE'\n")
    for callee, before, after in [('Macrophysics_simple','macro_in','macro_out'),
                                  ('microp_aero_ts','aero_in','aero_out'),
                                  ('mmicro_pcond',None,'micro_out')]:
        match = re.search(r'\bcall\s+'+callee+r'\s*\(', text, re.I)
        assert match, callee
        start = text.rfind('\n',0,match.start())+1
        depth=1; end=match.end()
        while depth:
            depth += (text[end]=='(')-(text[end]==')');end+=1
        end=text.index('\n',end)+1
        text=text[:start]+(dump(before) if before else '')+text[start:end]+dump(after)+text[end:]
    return text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapper(dump_constants=False,aerosol_constants=False,convection=False):
    # The declarations are extracted from the public argument section only.
    text = (WRF/'phys/module_mp_cammgmp_driver.F').read_text()
    declaration = text[text.index('    real,    intent(in) :: dt'):text.index('    ! Local variables')]
    declaration = re.sub(r'#if \( WRF_CHEM == 1 \).*?#endif', '', declaration, flags=re.S)
    three, two = [], []
    for line in declaration.splitlines():
        match = re.search(r'real,\s*dimension\((.*?)\).*?::\s*(\w+)', line, re.I)
        if match:
            (three if 'kms' in match[1] else two).append(match[2].lower())
    assert len(set(three)) == len(three) and len(three) > 35
    assert set(two) == set('ht qfx rliq2d xland snowh rainnc rainncv snownc snowncv sr'.split())
    header = (WRF.parents[0]/'WRF/phys/module_mp_cammgmp_driver.F').read_text()
    signature = header[header.index('  subroutine CAMMGMP(itimestep'):header.index('    !!-------------------------------------------------------- !', header.index('  subroutine CAMMGMP(itimestep'))]
    signature = re.sub(r'#if \( WRF_CHEM == 1 \).*?#endif', '', signature, flags=re.S)
    signature = re.sub(r'!.*', '', signature).replace('&', '').replace('\n', '')
    args = signature[signature.index('(')+1:signature.rindex(')')].split(',')
    args = [a.strip().lower() for a in args]
    dimensions = dict(ids=1, ide=2, jds=1, jde=2, kds=1, kde='nz+1',
                      ims=1, ime=1, jms=1, jme=1, kms=1, kme='nz+1',
                      its=1, ite=1, jts=1, jte=1, kts=1, kte='nz')
    call = ', &\n  '.join(f'{a}={dimensions.get(a,a)}' for a in args)
    old = Path('proofs/v034/camuw_oracle/camuw_oracle_driver.f90').read_text()
    useconst = old[old.index('  USE constituents,'):old.index('  IMPLICIT NONE')]
    init = old[old.index('  ALLOCATE(cnst_name(5)'):old.index('  restart = .FALSE.')].replace('(5)', '(12)')
    init += "\n  do m=1,7\n    call cnst_add(aernames(m),1._r8,cpair,0._r8,mm)\n  end do\n"
    initialize = '\n'.join(f'  {a}=0.' for a in three+two)
    dumps = '\n'.join(f"  call dump('{a}',{a}(1,:,1))" for a in three)
    dumps += '\n' + '\n'.join(f"  call dump('{a}',reshape({a},[1]))" for a in two)
    return f'''! Generated private original WRF CAM_INIT + CAMMGMP public caller.
program reference
  use shr_kind_mod, only: r8=>shr_kind_r8
  use module_mp_cammgmp_driver, only: CAMMGMP,CAMMGMP_INIT
  use module_configure, only: grid_config_rec_type
  use module_cam_support, only: pcnst_runtime,pcnst_mp
  use module_cam_esinti, only: esinti
  use modal_aero_initialize_data_phys, only: modal_aero_initialize_phys
  use modal_aero_data, only: cnst_name_cw,cnst_name_cw_mp
  use wv_saturation, only: polysvp,epsqs
{useconst}
  implicit none
  integer, parameter :: nz=24
  integer :: k,m,mm,ic,itimestep,mode
  real :: dt,accum_mode,aitken_mode,coarse_mode,temperature,rh
  real(r8) :: es,qs
  type(grid_config_rec_type) :: config_flags
  character(len=16), parameter :: aernames(7)=[character(len=16):: &
    'ACCUM_MASS','ACCUM_NUM','AITKEN_MASS','AITKEN_NUM', &
    'COARSE_MASS_1','COARSE_MASS_2','COARSE_NUM']
  real :: {','.join(a+'(1,nz+1,1)' for a in three)}
  real :: {','.join(a+'(1,1)' for a in two)}
  pcnst_runtime=12; pcnst_mp=12
  call esinti(epsilo,latvap,latice,rh2o,cpair,tmelt)
{init}
  allocate(cnst_name_cw(12),cnst_name_cw_mp(12))
  cnst_name_cw=cnst_name;cnst_name_cw_mp=cnst_name
  call modal_aero_initialize_phys
  call CAMMGMP_INIT(2,3,4,5,0,1,2,1,2,1,nz+1,1,1,1,1,1,nz+1,1,1,1,1,1,nz)
  {"call dump_cammg_constants" if dump_constants else ""}
  {"call dump_cammg_ndrop;call dump_cammg_aero;call dump_cammg_modal" if aerosol_constants else ""}
  config_flags%mp_physics=11
  config_flags%bl_pbl_physics=9
  config_flags%ra_lw_physics=3;config_flags%ra_sw_physics=3
  config_flags%cu_physics=0;config_flags%shcu_physics=0
  dt=18.
  do ic=1,{24 if convection else 18}
{initialize}
    accum_mode=1000.e6;aitken_mode=300.e6;coarse_mode=.2e6
    if(ic==16) then
      accum_mode=10.e6;aitken_mode=1.e6;coarse_mode=.01e6
    else if(ic==17) then
      accum_mode=3000.e6;aitken_mode=1000.e6;coarse_mode=5.e6
    else if(ic==18) then
      accum_mode=0.;aitken_mode=0.;coarse_mode=0.
    endif
    ht=125.;xland=1.;snowh=0.;qfx=1.e-5
    if(mod(ic,2)==0) xland=2.
    do k=1,nz+1
      z_at_w(1,k,1)=125.+400.*(k-1)
      p8w(1,k,1)=100000.*exp(-400.*(k-1)/8000.)
      z_sea_level(1,k,1)=z_at_w(1,k,1)+200.
      p_hyd(1,k,1)=100000.*exp(-(400.*(k-1)+200.)/8000.)
      mode=mod(ic-1,5)
      temperature=290.-3.*(k-1)
      if(mode==0) temperature=285.
      if(mode==1) temperature=240.
      if(mode==2) temperature=270.-1.5*(k-1)
      if(mode==3) temperature=280.-2.*(k-1)
      t_phy(1,k,1)=temperature
      pi_phy(1,k,1)=(p_hyd(1,k,1)/100000.)**(287./1004.5)
      th(1,k,1)=temperature/pi_phy(1,k,1)
      alt(1,k,1)=287.*temperature/p_hyd(1,k,1)
      es=polysvp(real(temperature,r8),0)
      qs=epsqs*es/(p_hyd(1,k,1)-(1._r8-epsqs)*es)
      rh=.98
      if(mode==4) rh=.50
      if(ic>=11 .and. ic<=15) rh=1.02
      qv_curr(1,k,1)=rh*qs/(1._r8-rh*qs)
      if(mode/=4) then
        qc_curr(1,k,1)=2.e-4
        qi_curr(1,k,1)=1.e-4
        if(mode==0) qi_curr(1,k,1)=0.
        if(mode==1) qc_curr(1,k,1)=0.
        nc3d(1,k,1)=1.e8;ni3d(1,k,1)=1.e5
        qr_curr(1,k,1)=3.e-4;qs_curr(1,k,1)=2.e-4
        nr3d(1,k,1)=1.e5;ns3d(1,k,1)=1.e4
      endif
      kvh3d(1,k,1)=5.;tke_pbl(1,k,1)=.5
      smaw3d(1,k,1)=1.;turbtype3d(1,k,1)=2.
    enddo
    rainnc=2.;snownc=.5
    {'''if(ic>=19)then
      config_flags%cu_physics=7;config_flags%shcu_physics=2
      if(ic==20)config_flags%shcu_physics=0
      if(ic==21)config_flags%cu_physics=0
      if(ic==22.or.ic==24)then
        config_flags%cu_physics=1;config_flags%shcu_physics=1
      endif
      icwmrdp3d=2.e-4;icwmrsh3d=1.e-4;shfrc3d=.05
      cmfmc3d=.004;cmfmc2_3d=.001
      dlf3d=2.e-7;dlf2_3d=1.e-7;rliq2d=1.e-7
    endif''' if convection else ''}
    do itimestep=1,2
      write(*,'(A,2I5)') 'BEGIN',ic,itimestep
      write(*,'(A)') 'INPUT'
{dumps}
      call CAMMGMP( &
  {call})
      write(*,'(A)') 'OUTPUT'
{dumps}
      write(*,'(A)') 'END'
      ! Match WRF phy_prep on the next public call, retaining the held fields.
      t_phy=th*pi_phy
    enddo
  enddo
contains
  subroutine dump(name,a)
    character(len=*),intent(in)::name
    real,intent(in)::a(:)
    write(*,'(A,*(1X,ES26.17E3))') trim(name),a
  end subroutine
end program
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('build',type=Path)
    ap.add_argument('--rebuild',choices=['zero','inf'])
    ap.add_argument('--observe',action='store_true')
    ap.add_argument('--constants',action='store_true')
    ap.add_argument('--aerosol-constants',action='store_true')
    ap.add_argument('--convection',action='store_true')
    args = ap.parse_args()
    args.build.mkdir(parents=True,exist_ok=True)
    driver=args.build/'reference.f90'
    driver.write_text(wrapper(args.constants,args.aerosol_constants,args.convection))
    if args.constants:
        driver.write_text(driver.read_text().replace('  use module_mp_cammgmp_driver, only:',
            '  use cldwat2m_micro, only: dump_cammg_constants\n  use module_mp_cammgmp_driver, only:'))
    if args.aerosol_constants:
        driver.write_text(driver.read_text().replace('  use module_mp_cammgmp_driver, only:',
            '  use ndrop,only:dump_cammg_ndrop\n  use microp_aero,only:dump_cammg_aero\n  use modal_aero_data,only:dump_cammg_modal\n  use module_mp_cammgmp_driver, only:'))
    stub=args.build/'stubs.f90'
    stub.write_text(Path('proofs/v034/camuw_oracle/wrf_runtime_stubs.f90').read_text() + '''
subroutine wrf_error_fatal(msg)
  character(len=*),intent(in)::msg
  write(0,'(A)') trim(msg)
  stop 2
end subroutine
''')
    flags=['-O2','-ffree-form','-ffree-line-length-none','-fconvert=big-endian','-frecord-marker=4',
           '-ffp-contract=off',f'-I{WRF}/phys',f'-I{WRF}/share',f'-I{WRF}/frame',f'-I{args.build}']
    objects=[WRF/'phys'/f'{m}.o' for m in MODULES]
    objects += [WRF/'share/module_model_constants.o',WRF/'frame/module_state_description.o']
    # All actual microphysics / activation bodies are rebuilt with defined init.
    if args.rebuild:
        flags += [f'-finit-real={args.rebuild}','-finit-integer=0','-finit-logical=false']
        rebuild = MODULES[:10]
        # Dependency order: aerosol data / init -> ndrop -> microp -> MG -> driver.
        rebuild = [MODULES[i] for i in [7,8,6,9,5,4,3,2,1,0]]
        for m in rebuild:
            src=WRF/'phys'/f'{m}.F'
            out=args.build/f'{m}.f90'
            prepared=subprocess.check_output(['cpp','-P','-traditional-cpp','-DWRF_CHEM=0','-DEM_CORE=1',str(src)],text=True)
            if args.observe and m=='module_mp_cammgmp_driver':
                prepared=observe(prepared)
            if args.constants and m=='module_cam_mp_cldwat2m_micro':
                from port import lines,declaration
                variables={}
                for _,s in lines(prepared[:prepared.lower().index('contains')]):
                    declaration(s,variables)
                fields=[name for name,v in variables.items() if v.kind=='f64' and not v.dims]
                getter="\nsubroutine dump_cammg_constants\n"+''.join(
                    f"write(*,'(A,1X,ES26.17E3)') 'INIT {name}',{name}\n" for name in fields)+"end subroutine\n"
                prepared=prepared.replace('  public :: ini_micro, mmicro_pcond,gamma',
                    '  public :: ini_micro, mmicro_pcond,gamma,dump_cammg_constants')
                prepared=prepared.replace('end module cldwat2m_micro',getter+'end module cldwat2m_micro')
            namespaces={'module_cam_mp_ndrop':('ndrop','ndrop'),
                        'module_cam_mp_microp_aero':('microp_aero','aero'),
                        'module_data_cam_mam_aero':('modal_aero_data','modal')}
            if args.aerosol_constants and m in namespaces:
                from port import lines,declaration
                module,namespace=namespaces[m];variables={}
                has_contains=re.search(r'^\s*contains\s*$',prepared,re.M|re.I)
                endmodule=re.search(r'^\s*end\s+module\s+'+module+r'\b',prepared,re.M|re.I)
                header_end=has_contains.start() if has_contains else endmodule.start()
                in_type=False
                for _,s in lines(prepared[:header_end]):
                    if re.match(r'type\s+\w+',s):in_type=True;continue
                    if s.startswith('end type'):in_type=False;continue
                    if not in_type:declaration(s,variables)
                getter=f'\nsubroutine dump_cammg_{namespace}\n'
                for name,v in variables.items():
                    if v.kind=='str':continue
                    if v.allocatable:getter+=f'if(allocated({name}))then\n'
                    getter+=f"write(*,'(A,1X,I0)') 'KIND {namespace}.{name}',{8 if v.kind=='f64' else 4 if v.kind=='f32' else 1 if v.kind=='bool' else 0}\n"
                    if v.dims:getter+=f"write(*,'(A,*(1X,I0))') 'SHAPE {namespace}.{name}',shape({name})\n"
                    value=f'merge(1._r8,0._r8,{name})' if v.kind=='bool' else f'real({name},r8)' if v.kind=='i32' else name
                    getter+=f"write(*,'(A,*(1X,ES26.17E3))') 'VALUE {namespace}.{name}',{value}\n"
                    if v.allocatable:getter+='endif\n'
                getter+='end subroutine\n'
                public=f'\npublic::dump_cammg_{namespace}\n'
                if has_contains:
                    prepared=prepared[:header_end]+public+prepared[header_end:]
                else:
                    prepared=prepared[:header_end]+public+'contains\n'+prepared[header_end:]
                endmodule=re.search(r'^\s*end\s+module\s+'+module+r'\b',prepared,re.M|re.I)
                prepared=prepared[:endmodule.start()]+getter+prepared[endmodule.start():]
            out.write_text(prepared)
            subprocess.run([str(FC),*flags,'-c',str(out),'-o',str(args.build/f'{m}.o')],cwd=args.build,check=True)
        objects=[args.build/p.name if p.stem in rebuild else p for p in objects]
    for src in [stub,driver]:
        subprocess.run([str(FC),*flags,'-c',str(src),'-o',str(src.with_suffix('.o'))],cwd=args.build,check=True)
    exe=args.build/'reference.exe'
    subprocess.run([str(FC),str(driver.with_suffix('.o')),str(stub.with_suffix('.o')),*map(str,objects),'-lmvec','-lm','-o',str(exe)],check=True)
    raw=args.build/'reference.txt'
    with raw.open('wb') as f:
        subprocess.run([str(exe)],cwd=args.build,stdout=f,check=True,timeout=120)
    manifest={'compiler':subprocess.check_output([str(FC),'--version'],text=True).splitlines()[0],
              'flags':flags,'initialization':args.rebuild or 'prebuilt-production',
              'sources':{str(WRF/'phys'/f'{m}.F'):sha(WRF/'phys'/f'{m}.F') for m in MODULES},
              'objects':{str(p):sha(p) for p in objects},'driver_sha256':sha(driver),'raw_sha256':sha(raw),
              'work':'nonchem CAM_INIT12/esinti/modal_aero/CAMMGMP_INIT;18 regimes;24levels;dt18;2steps;REAL4 public/r8 core'}
    (args.build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'raw':str(raw),'sha256':sha(raw),'bytes':raw.stat().st_size}))


if __name__=='__main__':
    main()
