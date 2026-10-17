"""Pristine MYNN hurricane FIRST-CALL oracle with original outer-caller guesses."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BUILD=Path('<USER_HOME>/wrf_gpu2_lanes/s2-oml/cold_flux')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
BUILD.mkdir(parents=True,exist_ok=True)
source=ROOT/'data/wrf_pristine/WRF/phys/module_sf_mynn.F'
shutil.copy2(source,BUILD/source.name)
shutil.copy2(ROOT/'proofs/v090/module_model_constants.f90',BUILD/'module_model_constants.f90')
driver=(ROOT/'proofs/v090/mynn_oracle_driver.f90').read_text()
driver=driver.replace('ustm=0.;','ustm=0.35;').replace('30(ES22.14','31(ES22.14').replace('ZNT(i)\n','ZNT(i),ustm(i)\n')
marker='  CALL SFCLAY1D_mynn('
bootstrap='''  ! Exact SFCLAY_mynn outer-caller bootstrap (module_sf_mynn.F:330-336).
  do i=1,ncol
    UST(i)=MAX(0.04*SQRT(U1D(i)*U1D(i)+V1D(i)*V1D(i)),0.001)
    MOL(i)=0.
    QSFC(i)=QV1D(i)/(1.+QV1D(i))
    qstar(i)=0.
  enddo
'''
assert marker in driver;driver=driver.replace(marker,bootstrap+marker)
(BUILD/'driver.f90').write_text(driver)
base=json.loads((ROOT/'proofs/s2_oml/fixtures/flux.json').read_text())
flags=['-O0','-ffp-contract=off','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1',
       '-finit-real=zero','-finit-integer=0','-finit-logical=false']


def build(extra=()):
    subprocess.run([FC,*flags,*extra,'module_model_constants.f90',source.name,'driver.f90','-o','oracle'],cwd=BUILD,check=True)


def oracle(rows,opt):
    text=f'{len(rows)} 1 1 {opt} 0 0 9000\n'+'\n'.join(' '.join(map(str,x)) for x in rows)+'\n'
    raw=subprocess.run([str(BUILD/'oracle')],input=text,text=True,capture_output=True,check=True).stdout
    return np.asarray([list(map(float,line.split()[1:])) for line in raw.splitlines() if line and not line.startswith('#')],np.float32)


result={};build()
for opt in (1,2):
    rows=np.asarray(base[f'mynn_{opt}']['inputs'],np.float32)
    expected=oracle(rows,opt);parts=[]
    for operand in (0,1,2,3,4,5,6,10,11,13,14,17,18,19,20,21,22):
        pair=[]
        for direction in (-np.inf,np.inf):
            varied=rows.copy();varied[:,operand]=np.nextafter(varied[:,operand],np.float32(direction))
            pair.append(np.abs(oracle(varied,opt)-expected))
        parts.append(np.max(pair,axis=0))
    result[f'mynn_{opt}']={'inputs':rows.tolist(),'output':expected.tolist(),
                          'input_ulp_bound':np.sum(parts,axis=0).tolist(),'ustm_in':.35,'first_timestep':True}
build(['-O2','-march=native','-ffp-contract=fast'])
for opt in (1,2):result[f'mynn_{opt}']['fma_output']=oracle(result[f'mynn_{opt}']['inputs'],opt).tolist()
(ROOT/'proofs/s2_oml/fixtures/cold_flux.json').write_text(json.dumps(result,indent=2)+'\n')
print('Original MYNN first-call bootstrap + Li stability seed: both hurricane options, 16columns')
