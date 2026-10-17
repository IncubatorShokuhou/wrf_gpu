"""Reproducible pristine LCZ/NUDAPT oracle, REAL4+REAL8, zero-init policy.

The staged WRF routines and noahmp_urban extract remain byte-for-byte pristine.
Only the previously qualified driver inputs change: LCZ IDs, NUDAPT geometry,
first-level thickness and table switches. All build outputs stay in lane storage.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
OUT = Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/oracle')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FILES = ('share/module_model_constants.F','frame/module_wrf_error.F',
         'phys/module_sf_urban.F','phys/module_bep_bem_helper.F','phys/module_sf_bem.F',
         'phys/module_sf_bep.F','phys/module_sf_bep_bem.F')
FLAGS = ['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1',
         '-ffpe-summary=none','-fallow-argument-mismatch','-finit-integer=0','-finit-real=zero',
         '-finit-logical=false']

def run(cmd,cwd):
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    subprocess.run(cmd,cwd=cwd,check=True,timeout=180)

OUT.mkdir(parents=True,exist_ok=True)
driver = (ROOT/'proofs/v040/urban_slucm/urban_oracle.f90').read_text()
driver = driver.replace('NSOIL, 1, 0, .FALSE.)','NSOIL, 1, 1, .FALSE.)')
driver = driver.replace('IVGTYP = 13','IVGTYP = 51 + MOD(r-1,12)\n    IF (IVGTYP(1,1)==62) IVGTYP=13')
driver = driver.replace('FRC_URB2D, UTYPE_URB2D, 0)','FRC_URB2D, UTYPE_URB2D, 1)')
driver = driver.replace('    FRC_URB2D = frc_in','''    FRC_URB2D = frc_in
    DZ8W(1,1,1) = 200.
    IF (MOD(r,2)==0) THEN
      MH_URB2D = 12.; HGT_URB2D = 12.; STDH_URB2D = 3.
      LP_URB2D = 0.3; LB_URB2D = 1.0
      IF (MOD(r,4)==0) LB_URB2D=0.2
      LF_URB2D(1,:,1) = (/ 0.1, 0.3, 0.5, 0.7 /)
    END IF
    CALL dumpin(r,'ivgtyp',REAL(IVGTYP(1,1)))
    CALL dumpin(r,'mh',MH_URB2D(1,1)); CALL dumpin(r,'hgt',HGT_URB2D(1,1))
    CALL dumpin(r,'stdh',STDH_URB2D(1,1)); CALL dumpin(r,'lp',LP_URB2D(1,1))
    CALL dumpin(r,'lb',LB_URB2D(1,1))''')
# zero_all already resets geometry for each regime. Output fields at step0 are
# not initialized outputs and are never included in numerical comparisons.
source_sha = {}
for name in FILES:
    source_sha[name] = hashlib.sha256((WRF/name).read_bytes()).hexdigest()
caller = ''.join((WRF/'phys/module_sf_noahmpdrv.F').read_text().splitlines(keepends=True)[2993:3783])
cal_source = (WRF/'phys/module_ra_gfdleta.F').read_text()
cal = cal_source[cal_source.index('SUBROUTINE CAL_MON_DAY'):]
cal = cal[:cal.index('END SUBROUTINE CAL_MON_DAY')+len('END SUBROUTINE CAL_MON_DAY')]
stub = '''logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_string(buf,n)
character(len=*) :: buf
integer :: n
end subroutine
subroutine wrf_dm_bcast_integer(buf,n)
integer :: buf(*),n
end subroutine
subroutine wrf_dm_bcast_real(buf,n)
real :: buf(*)
integer :: n
end subroutine
subroutine wrf_debug(level,msg)
integer :: level
character(len=*) :: msg
end subroutine
subroutine wrf_abort()
stop 1
end subroutine
'''
ids = 'MODULE NOAHMP_TABLES\nINTEGER :: ISURBAN_TABLE=13\n'
ids += '\n'.join(f'INTEGER :: LCZ_{i}_TABLE={50+i}' for i in range(1,12))+'\nEND MODULE\n'
for prec in ('fp32','fp64'):
    dest = OUT/prec
    dest.mkdir(exist_ok=True)
    flags = FLAGS+(['-fdefault-real-8','-fdefault-double-8'] if prec=='fp64' else [])
    for name in FILES:
        shutil.copy2(WRF/name,dest/Path(name).name)
    (dest/'stubs.f90').write_text(stub)
    (dest/'ids.f90').write_text(ids)
    (dest/'cal.F').write_text('MODULE module_ra_gfdleta\nCONTAINS\n'+cal+'\nEND MODULE\n')
    (dest/'caller.F').write_text('MODULE noahmp_urban_extract\nCONTAINS\n'+caller+'END MODULE\n')
    (dest/'driver.f90').write_text(driver)
    order = [Path(FILES[0]).name,Path(FILES[1]).name,'stubs.f90',*[Path(p).name for p in FILES[2:]],
             'ids.f90','cal.F','caller.F','driver.f90']
    for name in order:
        run([FC,*flags,'-c',name],dest)
    run([FC,*flags,'-o','oracle',*[Path(n).with_suffix('.o').name for n in order]],dest)
    for table in ('default','heat'):
        tabletext = (WRF/'run/URBPARM_LCZ.TBL').read_text()
        if table=='heat':
            tabletext = tabletext.replace('AHOPTION: 0','AHOPTION: 1').replace('ALHOPTION: 0','ALHOPTION: 1')
        (dest/'URBPARM_LCZ.TBL').write_text(tabletext)
        (OUT/f'URBPARM.TBL.{table}').write_text(tabletext)
        with (dest/f'{table}.txt').open('w') as dump:
            subprocess.run([str(dest/'oracle'),prec,str(ROOT/'proofs/v040/urban_slucm/regimes.txt')],
                cwd=dest,timeout=60,check=True,stdout=dump)
        run([sys.executable,str(ROOT/'proofs/v040/urban_slucm/dump_to_npz.py'),
             str(dest/f'{table}.txt'),str(OUT/f'slucm_{prec}_{table}.npz')],dest)
manifest={'wrf_commit':'f52c197ed39d12e087d02c50f412d90d418f6186','source_sha256':source_sha,
 'caller_sha256':hashlib.sha256(caller.encode()).hexdigest(),'driver_sha256':hashlib.sha256(driver.encode()).hexdigest(),
 'flags':FLAGS,'fp64_flags':['-fdefault-real-8','-fdefault-double-8'],
 'gate':'fp64:1e-10*ref_absmax;fp32:3*REAL4-REAL8 spread+4ulp;finite input and poststep outputs',
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
