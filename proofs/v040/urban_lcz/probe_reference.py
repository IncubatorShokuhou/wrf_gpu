"""Stage a pristine NUDAPT caller differential without changing WRF routines.

Run build_oracle.py first. Reuses that private pristine build's dependency objects. Rebuilds the
verbatim noahmp_urban caller with zero vs +Inf local initialization; all forcing
is identical. No GPU/JAX import. Run with the CPU/QUIET guards in AGENTS.md.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
OUT = Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/reference_probe')
BUILD = Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/oracle/fp32')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FLAGS = ['-O2', '-ffree-form', '-ffree-line-length-none', '-cpp', '-DEM_CORE=1',
         '-ffpe-summary=none', '-fallow-argument-mismatch', '-finit-integer=0','-finit-logical=false']

def run(cmd, cwd):
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'QUIET'
    subprocess.run(cmd, cwd=cwd, check=True, timeout=120, stdout=subprocess.DEVNULL)

OUT.mkdir(parents=True, exist_ok=True)
source = (WRF/'phys/module_sf_noahmpdrv.F').read_text().splitlines(keepends=True)
extract = ''.join(source[2993:3783])
assert extract == ''.join((BUILD/'caller.F').read_text().splitlines(keepends=True)[2:-1])
driver = (ROOT/'proofs/v040/urban_slucm/urban_oracle.f90').read_text()
driver = driver.replace('NSOIL, 1, 0, .FALSE.)', 'NSOIL, 1, 1, .FALSE.)')
driver = driver.replace('IVGTYP = 13', 'IVGTYP = 51 + MOD(r-1,11)')
driver = driver.replace('FRC_URB2D, UTYPE_URB2D, 0)', 'FRC_URB2D, UTYPE_URB2D, 1)')
driver = driver.replace('    FRC_URB2D = frc_in', '''    FRC_URB2D = 0.9
    DZ8W(1,1,1) = 200.
    MH_URB2D = 12.; HGT_URB2D = 12.; STDH_URB2D = 3.
    LP_URB2D = 0.3; LB_URB2D = 1.0
    LF_URB2D(1,:,1) = (/ 0.1, 0.3, 0.5, 0.7 /)''')
manifest = {'wrf_commit': 'f52c197ed39d12e087d02c50f412d90d418f6186',
            'caller_sha256': hashlib.sha256(extract.encode()).hexdigest(), 'flags': FLAGS, 'arms': {}}
objects = [p.name for p in BUILD.glob('*.o') if p.name not in ('driver.o','caller.o')]
for initialization in ('zero','inf'):
    dest = OUT/initialization
    dest.mkdir(exist_ok=True)
    for p in BUILD.glob('*.mod'):
        shutil.copy2(p, dest/p.name)
    for name in objects:
        target=dest/name
        if target.is_symlink(): target.unlink()
        target.symlink_to(BUILD/name)
    (dest/'caller.F').write_text('MODULE noahmp_urban_extract\nCONTAINS\n'+extract+'END MODULE\n')
    (dest/'driver.f90').write_text(driver)
    shutil.copy2(WRF/'run/URBPARM_LCZ.TBL', dest/'URBPARM_LCZ.TBL')
    run([FC,*FLAGS,f'-finit-real={initialization}','-c','caller.F'],dest)
    run([FC,*FLAGS,'-c','driver.f90'],dest)
    run([FC,*FLAGS,'-o','oracle','driver.o','caller.o',*objects],dest)
    with (dest/'dump.txt').open('w') as dump:
        subprocess.run([str(dest/'oracle'),'fp32',str(ROOT/'proofs/v040/urban_slucm/regimes.txt')], cwd=dest,
                       check=True,timeout=60,stdout=dump)
    run([sys.executable,str(ROOT/'proofs/v040/urban_slucm/dump_to_npz.py'),
         str(dest/'dump.txt'),str(OUT/f'{initialization}.npz')],dest)
    manifest['arms'][initialization] = {'caller_sha256': hashlib.sha256(extract.encode()).hexdigest(),
        'driver_sha256': hashlib.sha256(driver.encode()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
