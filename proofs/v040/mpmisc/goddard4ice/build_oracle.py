"""Build lane-local pristine Goddard4ice oracles; original bodies unmodified."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')
OLD=ROOT/'proofs/v018/mp_oracles/goddard4ice/oracle'
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--init',default='zero');parser.add_argument('--fma',action='store_true');parser.add_argument('--dt',type=float,default=90.);parser.add_argument('--dx',type=float,default=3000.);parser.add_argument('--steps',type=int,default=1);parser.add_argument('--land',type=int,default=1);parser.add_argument('--ulp',action='store_true');args=parser.parse_args()
    target=args.output.resolve();target.mkdir(parents=True,exist_ok=True)
    build=target/'build';build.mkdir(exist_ok=True)
    files=[WRF/'share/module_model_constants.F',WRF/'phys/module_mp_radar.F',WRF/'phys/module_mp_gsfcgce_4ice_nuwrf.F',OLD/'goddard4ice_stub_modules.f90',OLD/'goddard4ice_oracle_driver.f90']
    hashes={}
    for src in files:
        shutil.copy2(src,build/src.name);hashes[src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    driver=(build/files[4].name).read_text()
    driver=driver.replace('INTEGER :: k, case_id','INTEGER :: k, case_id, step_id, perturb, direction')
    vector_names=['th','qv','ql','qr','qi','qs','qg','qh','rho','pii','p','dz8w','z','w']
    perturb='  perturb=0\n  direction=1\n  IF (COMMAND_ARGUMENT_COUNT() >= 3) THEN\n    CALL GET_COMMAND_ARGUMENT(2,arg)\n    READ(arg,*) perturb\n    CALL GET_COMMAND_ARGUMENT(3,arg)\n    READ(arg,*) direction\n  ENDIF\n  SELECT CASE (perturb)\n'
    for index,key in enumerate(vector_names,1):perturb+=f'  CASE ({index})\n    {key}(1,:,1)=NEAREST({key}(1,:,1),REAL(direction))\n'
    perturb+='  END SELECT\n'
    driver=driver.replace('  CALL build_column(case_id)','  CALL build_column(case_id)\n'+perturb)
    driver=driver.replace('DT = 90.0',f'DT = {args.dt}')
    driver=driver.replace('DX = 3000.0',f'DX = {args.dx}')
    driver=driver.replace('xland = 1.0',f'xland = {args.land}.0')
    driver=driver.replace('  CALL gsfcgce_4ice_nuwrf(',f'  DO step_id=1,{args.steps}\n  CALL gsfcgce_4ice_nuwrf(')
    driver=driver.replace('                  1, xland, DX,','                  step_id, xland, DX,')
    driver=driver.replace('                  preci3d, precs3d, precg3d, prech3d, precr3d )','                  preci3d, precs3d, precg3d, prech3d, precr3d )\n  ENDDO')
    driver=driver.replace('  ! ---- call the real Goddard 4-ice microphysics ----', "  CALL dump_3d('TH_IN', th, .FALSE.)\n  CALL dump_3d('Z', z, .FALSE.)\n  CALL dump_3d('W', w, .FALSE.)\n  ! ---- call the real Goddard 4-ice microphysics ----")
    extras=['phys'+x for x in 'cedsmf']+['acphys'+x for x in 'cedsmf']+['preci3d','precs3d','precg3d','prech3d','precr3d']
    driver=driver.replace("  ! ---------------- dump everything (flat key=value) -----------------", '\n'.join(f"  CALL dump_3d('{key.upper()}', {key}, .FALSE.)" for key in extras)+"\n  CALL dump_3d('TH_OUT', th, .FALSE.)\n  ! ---------------- dump everything (flat key=value) -----------------")
    (build/files[4].name).write_text(driver)
    (target/'driver.f90').write_text(driver)
    # Exact original declarations and routine bodies, in a new getter-only module.
    text=files[2].read_text();decl=text[text.index('!  common /bt/'):text.index('CONTAINS')]
    routines=text[text.index('  SUBROUTINE consat_s'):text.index('!DIR$ ATTRIBUTES FORCEINLINE :: sgmap')]
    gamma=text[text.index(' real function gamma_toshi'):text.index(' SUBROUTINE Find_NaN_Inf_Double')]
    variables=[]
    from generate import statements,split
    for n,line in statements(decl):
        if re.match(r'^real\b',line):
            rhs=line.split('::',1)[1]
            variables.extend((v.strip(),31 if 'dimension' in line else None) for v in split(rhs))
    getter='subroutine dump_constants()\n integer k, itao\n itao=0\n call consat_s(itao)\n'
    for name,size in variables:
        if size:getter+=f" do k=1,{size}\n write(*,'(A,I0,A,ES23.15)') '{name}[',k,']=',{name}(k)\n enddo\n"
        else:getter+=f" write(*,'(A,ES23.15)') '{name}=',{name}\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_land_1=',gamma_toshi(min(15.e0,1000.e0/1500.e0+2.e0)+1.e0)\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_land_3=',gamma_toshi(min(15.e0,1000.e0/1500.e0+2.e0)+3.e0)\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_land_4=',gamma_toshi(min(15.e0,1000.e0/1500.e0+2.e0)+4.e0)\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_water_1=',gamma_toshi(min(15.e0,1000.e0/150.e0+2.e0)+1.e0)\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_water_3=',gamma_toshi(min(15.e0,1000.e0/150.e0+2.e0)+3.e0)\n"
    getter+=" write(*,'(A,ES23.15)') 'gamma_water_4=',gamma_toshi(min(15.e0,1000.e0/150.e0+2.e0)+4.e0)\n"
    getter+='end subroutine\n'
    (build/'constants.f90').write_text('module goddard_constant_oracle\n'+decl+'contains\n'+routines+'\n'+gamma+'\n'+getter+'end module\nprogram dump\nuse goddard_constant_oracle\ncall dump_constants()\nend program\n')
    flags=['-O2','-cpp','-ffree-form','-ffree-line-length-none','-ffpe-summary=none','-fallow-argument-mismatch',f'-finit-real={args.init}','-finit-integer=0','-finit-logical=false','-ffp-contract=off']
    physics_flags=flags[:-1]+['-ffp-contract=fast','-march=native'] if args.fma else flags
    run=lambda cmd:subprocess.run(cmd,cwd=build,check=True)
    for src in [files[3],*files[:3],files[4]]:run([FC,*(physics_flags if src==files[2] else flags),'-c',src.name])
    run([FC,*flags,'-o','oracle',*[src.stem+'.o' for src in files]])
    import importlib.util
    spec=importlib.util.spec_from_file_location('goddard_dump',OLD/'goddard4ice_dump_to_json.py')
    parser_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser_module)
    variants=[(0,1)]+([(i,d) for i in range(1,len(vector_names)+1) for d in (-1,1)] if args.ulp else [])
    for case in range(1,7):
        for index,direction in variants:
            stem=f'case{case}' if not index else f'case{case}_v{index}_{"p" if direction>0 else "m"}'
            data=subprocess.check_output([str(build/'oracle'),str(case),str(index),str(direction)],cwd=build,text=True)
            (target/f'{stem}.txt').write_text(data)
            parser_module.main(str(target/f'{stem}.txt'),str(target/f'{stem}.json'),'WRF pristine gsfcgce_4ice_nuwrf')
    run([FC,*flags,'-o','constants','constants.f90'])
    const=subprocess.check_output([str(build/'constants')],cwd=build,text=True)
    (target/'constants.txt').write_text(const)
    constants={}
    for line in const.splitlines():
        if '=' not in line:continue
        key,value=line.split('=',1);value=float(value)
        if '[' in key:
            name,index=key.split('['); index=int(index[:-1]);constants.setdefault(name,[0.]*31)[index-1]=value
        else:constants[key]=value
    (target/'constants.json').write_text(json.dumps(constants,indent=2))
    (target/'provenance.json').write_text(json.dumps({'sources':hashes,'flags':flags,'physics_flags':physics_flags,'caller':{'dt':args.dt,'dx':args.dx,'steps':args.steps,'land':args.land},'ulp_vectors':vector_names,'driver_sha256':hashlib.sha256(driver.encode()).hexdigest(),'original_source_untouched':True,'constant_bodies_verbatim':True},indent=2))
    # Exact path under caller-owned lane directory, never a variable rm -rf.
    shutil.rmtree(build)
if __name__=='__main__':main()
