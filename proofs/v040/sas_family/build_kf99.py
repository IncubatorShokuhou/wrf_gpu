"""Build unmodified old-KF, REAL4/REAL8, compiler-defined init sensitivity.

Caller owns QUIET check, CPU pin/nice and timeout. No product flags change.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

work = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent
source = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_kf.F')
compiler = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
work.mkdir(parents=True, exist_ok=True)
tmp = work/'tmp'
tmp.mkdir(exist_ok=True)
env = dict(os.environ, TMPDIR=str(tmp))
manifest = {'source':str(source), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'driver_sha256':hashlib.sha256((here/'kf99_oracle.F90').read_bytes()).hexdigest(), 'arms':{}}
for mode in ('r4','r8'):
    for init in ('zero','snan'):
        build = work/f'build_{mode}_{init}'
        build.mkdir(exist_ok=True)
        shutil.copyfile(source,build/source.name)
        shutil.copyfile(here/'kf99_oracle.F90',build/'kf99_oracle.F90')
        # Logging interface only; old KFPARA formats a message into this public buffer.
        stub = (here.parents[1]/'v060/oracle/module_wrf_error.f90').read_text()
        stub = stub.replace('  IMPLICIT NONE\nCONTAINS',
                            '  IMPLICIT NONE\n  CHARACTER(LEN=512) :: wrf_err_message\nCONTAINS')
        (build/'module_wrf_error.f90').write_text(stub)
        flags = ['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none',
                 '-cpp','-DEM_CORE=1','-DNMM_CORE=0',f'-finit-real={init}','-finit-integer=0','-finit-logical=false']
        if mode == 'r8': flags += ['-fdefault-real-8','-fdefault-double-8']
        def run(args):
            result = subprocess.run([compiler]+args,cwd=build,env=env,capture_output=True)
            if result.returncode:
                print(result.stdout.decode(), result.stderr.decode(), flush=True)
            result.check_returncode()
            return result
        run(flags+['-c','module_wrf_error.f90'])
        run(flags+['-c',source.name])
        run(flags+['-c','kf99_oracle.F90'])
        run(['-O2','-o','kf99_oracle','module_wrf_error.o','module_cu_kf.o','kf99_oracle.o'])
        rec = {'flags':flags,'cases':{}}
        for case in range(1,6):
            result = subprocess.run([str(build/'kf99_oracle'),str(case)],cwd=build,env=env,
                                    check=True,capture_output=True)
            out = work/f'case{case}_{mode}_{init}.txt'
            out.write_bytes(result.stdout)
            rec['cases'][str(case)] = {'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
                                      'stderr':result.stderr.decode()}
        manifest['arms'][f'{mode}_{init}'] = rec
        (work/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(mode,init,'5 original cases complete',flush=True)
