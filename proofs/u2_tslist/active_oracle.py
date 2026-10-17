"""Compile verbatim pristine calc_p8w plus its literal active CLW loop.

The coupled Swiss columns are clear; this supplemental source oracle activates
all five condensate species on all 44 real pressure/height levels. Not a claim
of a cloudy coupled forecast. Fortran is a private lane build, no shared writes.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
source_path = repo / 'data/wrf_pristine/WRF/share/wrf_timeseries.F'
source = source_path.read_text()
routine = source[source.index('SUBROUTINE calc_p8w('):source.index('END SUBROUTINE calc_p8w')+len('END SUBROUTINE calc_p8w')]
clw_loop = source[source.index('         clw=0.'):source.index('         clw = clw / g')+len('         clw = clw / g')]
stub = '''module module_model_constants
 real, parameter :: g=9.81
end module
module module_domain
 type domain
  real :: phb(2,45,1),ph_2(2,45,1),p(2,44,1),pb(2,44,1),fnm(44),fnp(44)
  real :: moist(2,44,1,5)
  integer :: sm32=1,em32=45
 end type
end module
'''
driver = '''program active_clw
 use module_domain
 use module_model_constants
 type(domain) :: grid
 real :: p8w(45),clw
 integer :: ix,iy,k,mm
 integer,parameter :: num_moist=5,P_QC=1,P_QR=2,P_QI=3,P_QS=4,P_QG=5
 open(41,file='input.bin',access='stream',form='unformatted')
 read(41) grid%p,grid%pb,grid%ph_2,grid%phb,grid%fnm,grid%fnp,grid%moist
 close(41)
 open(42,file='output.bin',access='stream',form='unformatted',status='replace')
 iy=1
 do ix=1,2
  call calc_p8w(grid,ix,iy,p8w,1,45)
'''+clw_loop+'''
  write(42) p8w,clw
 enddo
 close(42)
end program
'''
(out/'oracle.f90').write_text(stub+'\n'+routine+'\n'+driver)
with np.load(repo/'proofs/u2_tslist/oracle/columns.npz') as raw:
    inp = {}
    for key in ('P','PB','PH','PHB'):
        inp[key] = raw[key][0, :, :, 0, 0, None]
    dnw = np.diff(raw['ZNW'])
    dn = np.float32(.5)*(dnw[:-1]+dnw[1:])
    inp['fnm'] = np.r_[np.float32(0), np.float32(.5)*dnw[:-1]/dn]
    inp['fnp'] = np.r_[np.float32(0), np.float32(.5)*dnw[1:]/dn]
q = np.empty((2,44,1,5),np.float32)
for species in range(5):
    q[:,:,:,species] = (np.float32(species+1)*np.float32(2.e-5)
                       * (np.float32(1)+np.arange(44,dtype=np.float32)[None,:,None]/np.float32(44)))
inp['moist'] = q
with (out/'input.bin').open('wb') as file:
    for value in inp.values():
        file.write(np.asarray(value,np.float32).tobytes(order='F'))
fc = os.environ.get('FC','<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran')
subprocess.run([fc,'-O2','-ffp-contract=off','-ffree-line-length-none','oracle.f90','-o','oracle'],cwd=out,check=True,timeout=60)
subprocess.run([str(out/'oracle')],cwd=out,check=True,timeout=30)
result = np.fromfile(out/'output.bin',np.float32).reshape(2,46)
target = out/'active_clw.npz'
np.savez_compressed(target, **inp,p8w=result[:,:45],clw=result[:,45])
manifest = {'source':str(source_path), 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'oracle_source_sha256':hashlib.sha256((out/'oracle.f90').read_bytes()).hexdigest(),
            'fixture_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'compiler':fc, 'flags':['-O2','-ffp-contract=off','-ffree-line-length-none'],
            'active_species':5,'active_levels':44,'clw_kg_m2':result[:,45].tolist(),
            'scope':'pristine literal source, constructed active condensate on real Swiss pressure/height columns'}
(out/'active_clw.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
