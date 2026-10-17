"""Extract pristine routines verbatim into a small independent CPU oracle.

Run with --source <USER_HOME>/src/wrf_pristine/WRF --output <lane-dir>.
Does not edit/build in the pristine tree. GPL/MIT notices remain in source tree;
the generated source is a local test artifact, never a runtime dependency.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    mp = (source/'phys/module_mp_thompson.F').read_text()
    radar_path = source/'phys/module_mp_radar.F'
    # Keep original declarations/constants and only routines actually called.
    header = mp[:mp.lower().index('\n      contains')]
    def routine(start, end):
        a = mp.lower().index(start.lower())
        b = mp.lower().index(end.lower(), a)+len(end)
        return mp[a:b]
    calc = routine('      subroutine calc_refl10cm ', '      end subroutine calc_refl10cm')
    gamma = routine('      REAL FUNCTION GAMMLN(XX)', '      END FUNCTION GAMMLN')
    wgamma = routine('      REAL FUNCTION WGAMMA(y)', '      END FUNCTION WGAMMA')
    init = '''
      subroutine oracle_init
      integer n
      cre(2)=mu_r+1.
      cre(3)=bm_r+mu_r+1.
      cre(4)=bm_r*2.+mu_r+1.
      do n=2,4
       crg(n)=WGAMMA(cre(n))
      enddo
      org2=1./crg(2)
      obmr=1./bm_r
      cse(1)=bm_s+1.
      cse(3)=bm_s*2.
      oams=1./am_s
      obms=1./bm_s
      ocms=oams**obms
      cge(2,:)=mu_g+1.
      cge(3,:)=bm_g+mu_g+1.
      cge(4,:)=bm_g*2.+mu_g+1.
      do n=2,4
       cgg(n,:)=WGAMMA(cge(n,1))
      enddo
      ogg2=1./cgg(2,1)
      obmg=1./bm_g
      call radar_init
      end subroutine
'''
    # Simple stdin boundary: N columns, K levels, then 10 REAL arrays per column.
    driver = '''
program oracle
 use module_mp_thompson
 implicit none
 integer nz,ncol,i,k,v
 real,allocatable::a(:,:),z(:)
 read(*,*)ncol,nz
 allocate(a(nz,10),z(nz))
 call oracle_init
 do i=1,ncol
  do v=1,10
   read(*,*)a(:,v)
  enddo
  call calc_refl10cm(a(:,1),a(:,2),a(:,3),a(:,4),a(:,5),a(:,6), &
                    a(:,7),a(:,8),a(:,9),a(:,10),z,1,nz,1,1,nz)
  do k=1,nz
   write(*,'(ES24.16)')max(-35.,z(k))
  enddo
 enddo
end program
'''
    stub = '''
module module_wrf_error
contains
 subroutine wrf_debug(n,s)
 integer n
 character(*)s
 end subroutine
end module
module module_model_constants
 real,parameter::RE_QC_BG=2.5e-6,RE_QI_BG=10e-6,RE_QS_BG=10e-6
end module
'''
    (output/'stub.f90').write_text(stub)
    (output/'radar.F90').write_text(radar_path.read_text())
    (output/'thompson.F90').write_text(header+'\nCONTAINS\n'+init+calc+'\n'+gamma+'\n'+wgamma+'\nend module\n')
    (output/'driver.f90').write_text(driver)
    compiler = os.environ.get('FC', 'gfortran')
    cmd = [compiler,'-cpp','-ffree-form','-ffree-line-length-none','-O2','-fno-fast-math','-ffp-contract=off',
           'stub.f90','radar.F90','thompson.F90','driver.f90','-o','oracle']
    subprocess.run(cmd, cwd=output, check=True)
    manifest = {'sources': {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source/'phys/module_mp_thompson.F', radar_path)},
                'calc_refl10cm_sha256': hashlib.sha256(calc.encode()).hexdigest(),
                'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
                'command':cmd, 'scope':'verbatim calc_refl10cm + radar module; same REAL constants; initialization of used moments only; no model stepping'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output/'oracle'


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)
