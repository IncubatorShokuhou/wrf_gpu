"""Two pristine NSSL calls on real Swiss columns, including inactive CCI.

The existing audited legacy19/21 oracle caller is reused; original module
bodies stay byte-identical. Ni is populated with observed ice number to make
preservation nonvacuous although these one-moment packages do not own it.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

ROOT=Path(__file__).resolve().parents[2]
CASE=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_06:00:00')
NAMES=('TH','QV','QC','QR','QI','QS','QH','QHL','CCW','CRW','CCI','CSW','CHW','CHL','CN','VHW','VHL','PII','P','DZ','W','DN')


def main(build,dest):
    assert os.environ['JAX_PLATFORMS']=='cpu'
    spec=importlib.util.spec_from_file_location('legacy',ROOT/'proofs/s2_mpmore/nssl1mom/build_reference.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    driver=legacy.make_driver(ROOT/'proofs/v034/oracle/nssl2mom').replace('KX  = 40','KX  = 44')
    driver=driver.replace('  INTEGER :: variant','  INTEGER :: variant, step\n  CHARACTER(LEN=1024) :: input_file, output_file')
    anchor='  ! save inputs'
    assert driver.count(anchor)==1
    read='\n'.join('  READ(87) '+name for name in NAMES)
    driver=driver.replace(anchor,'''  CALL GET_COMMAND_ARGUMENT(4,input_file)
  CALL GET_COMMAND_ARGUMENT(5,output_file)
  OPEN(87,file=TRIM(input_file),access='stream',form='unformatted',status='old')
'''+read+'''
  CLOSE(87)
  DT=18.0
  OPEN(88,file=TRIM(output_file),access='stream',form='unformatted',status='replace')
'''+anchor)
    driver=driver.replace('  CALL nssl_2mom_driver(', '  DO step=1,2\n  itimestep=step\n  CALL nssl_2mom_driver(')
    anchor='  ! ---------------- dump everything'
    at=driver.index(anchor)
    driver=driver[:at]+"  WRITE(88) "+','.join(NAMES)+"\n  END DO\n  CLOSE(88)\n  STOP\n"+driver[at:]
    build.mkdir(parents=True)
    source=ROOT/'data/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F'
    (build/'module.F').write_bytes(source.read_bytes())
    (build/'stubs.f90').write_bytes((ROOT/'proofs/v034/oracle/nssl2mom/nssl2mom_stub_modules.f90').read_bytes())
    (build/'caller.F90').write_text(driver)
    (build/'namelist.input').write_text('')
    fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-ffp-contract=off','-finit-real=zero','-finit-integer=0','-finit-logical=false']
    for name in ('stubs.f90','module.F','caller.F90'):
        subprocess.run([fc,*flags,'-c',name],cwd=build,check=True,timeout=180)
    subprocess.run([fc,'stubs.o','module.o','caller.o','-o','oracle'],cwd=build,check=True,timeout=60)
    with Dataset(CASE) as ds:
        get=lambda n:np.asarray(ds[n][0],np.float32)
        mass={n:get(v) for n,v in zip(NAMES[1:7],('QVAPOR','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP'))}
        points=[]
        for n in mass:
            y,x=np.unravel_index(np.argmax(mass[n][:,2:-2,2:-2].max(axis=0)),mass[n].shape[1:]-np.array([4,4]))
            point=(int(y)+2,int(x)+2)
            if point not in points:points.append(point)
        ys,xs=np.asarray(points).T
        pick=lambda a:a[:,ys,xs].T.copy()
        data={n:pick(a) for n,a in mass.items()}
        data['THM']=pick(get('THM')+np.float32(300));data['P']=pick(get('P')+get('PB'))
        data['PH']=pick(get('PH')+get('PHB'));data['W']=pick(get('W')[:-1]);data['CCI']=pick(get('QNICE'))
    data['TH']=data['THM']/(np.float32(1)+np.float32(461.6/287)*data['QV'])
    data['PII']=(data['P']/np.float32(1e5))**np.float32(2/7)
    temperature=data['TH']*data['PII']
    data['DN']=(data['P']/(np.float32(287)*temperature*(np.float32(1)+np.float32(461.6/287)*data['QV'])))*(np.float32(1)+data['QV'])
    data['DZ']=np.maximum(np.diff(data['PH'],axis=1)/np.float32(9.81),np.float32(1))
    for n in NAMES:
        if n not in data:data[n]=np.zeros_like(data['TH'])
    result={**data}
    for variant in (19,21):
        outputs=[]
        for c in range(len(points)):
            np.stack([data[n][c] for n in NAMES]).astype(np.float32).tofile(build/'input.bin')
            subprocess.run([str(build/'oracle'),'1','none',str(variant),str(build/'input.bin'),str(build/'output.bin')],cwd=build,check=True,timeout=30,stdout=subprocess.DEVNULL)
            outputs.append(np.fromfile(build/'output.bin',np.float32).reshape(2,len(NAMES),44))
        result[f'v{variant}']=np.stack(outputs,axis=1)
        assert np.isfinite(result[f'v{variant}']).all()
        assert np.array_equal(result[f'v{variant}'][:,:,NAMES.index('CCI'),:],np.broadcast_to(data['CCI'],(2,)+data['CCI'].shape))
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'swiss.npz',**result)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(source_sha256=sha(source),caller_sha256=sha(build/'caller.F90'),input=str(CASE),input_sha256=sha(CASE),
                  fixture_sha256=sha(dest/'swiss.npz'),fields=NAMES,points_y_x=points,steps=2,dt=18.,flags=flags,
                  nonzero_cci=int(np.count_nonzero(data['CCI'])),scope='two original MP calls; observed non-owned Ni tests preservation; not a forecast',
                  source_files={n:sha(ROOT/'data/wrf_pristine/WRF'/n) for n in ('Registry/Registry.EM_COMMON','share/module_check_a_mundo.F','phys/module_pbl_driver.F','phys/module_microphysics_driver.F','dyn_em/solve_em.F')})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
