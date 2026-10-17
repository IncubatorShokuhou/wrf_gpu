"""Build a lane-private pristine-WRF REAL4 helper oracle with bounds checking."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

wrf, build = map(Path, sys.argv[1:3])
build.mkdir(parents=True, exist_ok=True)
temp=build/'compiler_tmp';temp.mkdir(exist_ok=True)
compiler_env={**os.environ,'TMPDIR':str(temp)}
here = Path(__file__).resolve().parent
src = wrf/'phys/module_ra_goddard.F'
text = src.read_text()
marker = '  public :: init_goddardrad, goddardrad'
assert text.count(marker) == 1
text = text.replace(marker, '  public :: read_lut_nc, opt_cloud, sounding_interp, ozone_interp\n'+marker)
(build/'module_ra_goddard.F').write_text(text)
shutil.copy2(wrf/'phys/module_checkerror.F', build)
shutil.copy2(here.parent/'v013/oracle/radiation_lw/goddard_lw_stubs.f90', build/'stubs.f90')
shutil.copy2(here/'helpers_driver.f90', build/'driver.f90')
shutil.copy2(wrf/'run/BROADBAND_CLOUD_GODDARD.bin', build)
fc = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags = ['-O2', '-ffree-form', '-ffree-line-length-none', '-cpp', '-DRWORDSIZE=4',
         '-DNO_IEEE_MODULE', '-fcheck=bounds', '-ffpe-summary=none',
         '-fallow-argument-mismatch', '-std=legacy', '-fconvert=big-endian', '-frecord-marker=4']
cmd = [fc, *flags, 'stubs.f90', 'module_checkerror.F', 'module_ra_goddard.F', 'driver.f90', '-o', 'helpers']
subprocess.run(cmd, cwd=build, env=compiler_env, check=True)
with (build/'helpers.txt').open('w') as output:
    subprocess.run([str(build/'helpers')], cwd=build, stdout=output, check=True)
equator = subprocess.run([str(build/'helpers'), 'equator'], cwd=build, text=True, capture_output=True)
(build/'equator.txt').write_text(equator.stdout+equator.stderr)
manifest = {
    'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
    'shim_sha256': hashlib.sha256((build/'module_ra_goddard.F').read_bytes()).hexdigest(),
    'driver_sha256': hashlib.sha256((here/'helpers_driver.f90').read_bytes()).hexdigest(),
    'command': cmd,
    'equator_returncode': equator.returncode,
    'affinity': sorted(os.sched_getaffinity(0)),
    'physics_changes': 'none; visibility-only shim',
    'cache': 'none; no JAX; lane-private Fortran build',
}
(build/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
