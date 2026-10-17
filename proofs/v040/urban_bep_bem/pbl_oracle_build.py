"""WRF YSU/BouLac urban-consumer oracles: prescribed active source coefficients.

Physics source is unmodified. R4 uses the pristine ccpp_kind_types; R8 changes
only its precision selection, alongside -fdefault-real-8, for roundoff diagnosis.
Inputs are operator stress regimes derived from the v060 sounding drivers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')


def make_driver(scheme):
    old=HERE.parents[1]/'v060/oracle'/f'{scheme}_oracle_driver.f90'
    source=old.read_text()
    if scheme=='ysu':
        declarations='''  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: A_U,A_V,A_T,A_Q,A_E,B_U,B_V,B_T,B_Q,B_E
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: DLG_BEP,DL_U_BEP,SF_BEP,VL_BEP
  REAL, DIMENSION(ims:ime,jms:jme) :: FRC_URB2D
'''
        source=source.replace('  CHARACTER(LEN=256) :: errmsg',declarations+'  CHARACTER(LEN=256) :: errmsg')
        source=source.replace('           idiff=idiff,flag_bep=flag_bep, &','''           idiff=idiff,flag_bep=flag_bep, &
           a_u_bep=A_U,a_v_bep=A_V,a_t_bep=A_T,a_q_bep=A_Q,a_e_bep=A_E, &
           b_u_bep=B_U,b_v_bep=B_V,b_t_bep=B_T,b_q_bep=B_Q,b_e_bep=B_E, &
           dlg_bep=DLG_BEP,dl_u_bep=DL_U_BEP,sf_bep=SF_BEP,vl_bep=VL_BEP,frc_urb2d=FRC_URB2D, &''')
    source=source.replace('  INTEGER :: case_id','  INTEGER :: case_id,urban_k').replace('  INTEGER :: errflg, case_id, topdown, idiff','  INTEGER :: errflg, case_id, topdown, idiff,urban_k')
    source=source.replace('  flag_bep = .FALSE.','  flag_bep = .TRUE.')
    # Preserve sounding inputs, add controlled implicit/explicit/geometry terms.
    init='''  A_U=0.;A_V=0.;A_T=0.;A_Q=0.;A_E=0.;B_U=0.;B_V=0.;B_T=0.;B_Q=0.;B_E=0.
  DLG_BEP=0.;DL_U_BEP=0.;SF_BEP=1.;VL_BEP=1.
  FRC_URB2D=0.15*case_id
  IF (case_id==5) FRC_URB2D=0.
  DO urban_k=1,KX
    A_U(1,urban_k,1)=-FRC_URB2D(1,1)*.002*EXP(-REAL(urban_k-1)/2.)
    A_V(1,urban_k,1)=A_U(1,urban_k,1)*.7
    B_T(1,urban_k,1)=FRC_URB2D(1,1)*.03*EXP(-REAL(urban_k-1)/2.)
    B_Q(1,urban_k,1)=FRC_URB2D(1,1)*1.e-6*EXP(-REAL(urban_k-1)/2.)
    B_E(1,urban_k,1)=FRC_URB2D(1,1)*.002*EXP(-REAL(urban_k-1)/2.)
    SF_BEP(1,urban_k,1)=1.-.25*FRC_URB2D(1,1)*EXP(-REAL(urban_k-1)/2.)
    VL_BEP(1,urban_k,1)=1.-.3*FRC_URB2D(1,1)*EXP(-REAL(urban_k-1)/2.)
    DLG_BEP(1,urban_k,1)=DZ(1,urban_k,1)/2.
    IF (urban_k<=4) DL_U_BEP(1,urban_k,1)=20.
  ENDDO
  SF_BEP(1,1,1)=1.
'''
    for field in ['A_U','A_V','A_T','A_Q','A_E','B_U','B_V','B_T','B_Q','B_E','DLG_BEP','DL_U_BEP','SF_BEP','VL_BEP']:
        init+=f"  CALL dump_col('URBAN_{field}',{field})\n"
    init+="  WRITE(*,'(A,ES25.17)') 'FRC_URB=',FRC_URB2D(1,1)\n"
    marker='  CALL '+('ysu(' if scheme=='ysu' else 'boulac(')
    assert source.count(marker)==1
    source=source.replace(marker,init+'\n'+marker)
    return source


def parse(path):
    scalars={};columns={}
    for line in path.read_text().splitlines():
        m=re.fullmatch(r'([A-Z0-9_]+)\[(\d+)\]=\s*(\S+)',line)
        if m:
            name,index,value=m.groups();index=int(index)
            assert index not in columns.setdefault(name,{})
            columns[name][index]=float(value)
        else:
            m=re.fullmatch(r'([A-Z0-9_]+)=\s*(.*)',line)
            if m:
                try:scalars[m[1]]=float(m[2])
                except ValueError:scalars[m[1]]=m[2]
    assert scalars['KX']==32
    values={name:[vals[k] for k in range(1,34 if name=='PDI' else 33)] for name,vals in columns.items()}
    assert all(np.isfinite(v).all() for v in values.values())
    assert 'URBAN_A_U' in values
    return {'scalars':scalars,'columns':values}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);args=parser.parse_args()
    assert os.environ['JAX_PLATFORMS']=='cpu';assert not Path('/tmp/wrf_gpu2_quiet').exists()
    args.build.mkdir(parents=True,exist_ok=True)
    flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-ffpe-summary=none']
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran';manifest={'source_hashes':{},'flags':flags,'fixtures':{}}
    paths=['phys/ccpp_kind_types.f90','phys/physics_mmm/bl_ysu.F90','phys/module_bl_ysu.F','phys/module_bl_boulac.F']
    for path in paths:manifest['source_hashes'][path]=hashlib.sha256((WRF/path).read_bytes()).hexdigest()
    for prec in ['fp32','fp64']:
        build=args.build/prec;build.mkdir(exist_ok=True)
        pflags=flags+(['-fdefault-real-8','-fdefault-double-8'] if prec=='fp64' else [])
        for path in paths:shutil.copyfile(WRF/path,build/Path(path).name)
        if prec=='fp64':
            p=build/'ccpp_kind_types.f90';p.write_text(p.read_text().replace('selected_real_kind(6)','selected_real_kind(15)'))
        for file in ['ccpp_kind_types.f90','bl_ysu.F90','module_bl_ysu.F','module_bl_boulac.F']:
            subprocess.run([fc,*pflags,'-c',file],cwd=build,check=True,timeout=60)
        for scheme in ['ysu','boulac']:
            src=build/f'{scheme}_oracle.f90';src.write_text(make_driver(scheme))
            subprocess.run([fc,*pflags,'-o',scheme,src.name,'ccpp_kind_types.o','bl_ysu.o','module_bl_ysu.o','module_bl_boulac.o'],cwd=build,check=True,timeout=60)
            for case in range(1,7):
                raw=build/f'{scheme}_{case}.txt'
                with raw.open('w') as out:subprocess.run([str(build.resolve()/scheme),str(case)],cwd=build,stdout=out,check=True,timeout=30)
                data=parse(raw);file=f'{scheme}_{prec}_{case}.json';(args.build/file).write_text(json.dumps(data,indent=2)+'\n')
                manifest['fixtures'][file]=hashlib.sha256((args.build/file).read_bytes()).hexdigest()
    (args.build/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS',len(manifest['fixtures']),'urban PBL oracle fixtures')


if __name__=='__main__':main()
