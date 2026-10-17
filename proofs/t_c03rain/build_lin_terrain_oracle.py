"""Build an unchanged pristine Lin body; translate caller columns above terrain."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF/phys')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FLAGS = ['-O2', '-ffree-form', '-ffree-line-length-none', '-cpp', '-ffpe-summary=none',
         '-fno-range-check', '-finit-real=zero', '-finit-integer=0', '-finit-logical=false']


def build(out: Path) -> None:
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    build = out / 'build'
    build.mkdir(parents=True, exist_ok=True)
    for name in ('module_mp_radar.F', 'module_mp_lin.F'):
        shutil.copyfile(WRF / name, build / name)
    shutil.copyfile(ROOT / 'proofs/v060/oracle/module_wrf_error_lin.f90', build / 'module_wrf_error.f90')
    driver = (ROOT / 'proofs/v060/oracle/lin_oracle_driver.f90').read_text()
    driver = driver.replace('  CALL build_column(case_id)', '''  CALL build_column(case_id)
  ! Only the caller changes: translate the absolute-height column and HT together.
  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN
    CALL GET_COMMAND_ARGUMENT(2, arg)
    READ(arg,*) ht(1,1)
    z = z + ht(1,1)
  END IF''')
    driver = driver.replace("  WRITE(*,'(A,I0)') 'CASE=', case_id", "  WRITE(*,'(A,ES23.15)') 'HT=', ht(1,1)\n  WRITE(*,'(A,I0)') 'CASE=', case_id")
    (build / 'driver.f90').write_text(driver)
    commands = []
    for name in ('module_wrf_error.f90', 'module_mp_radar.F', 'module_mp_lin.F', 'driver.f90'):
        cmd = [FC, *FLAGS, '-c', name]
        subprocess.run(cmd, cwd=build, check=True, timeout=90)
        commands.append(cmd)
    cmd = [FC, '-o', 'oracle', 'module_wrf_error.o', 'module_mp_radar.o', 'module_mp_lin.o', 'driver.o']
    subprocess.run(cmd, cwd=build, check=True, timeout=30)
    commands.append(cmd)
    for case in range(1, 7):
        for ht in (0, 1500):
            result = subprocess.run([str(build / 'oracle'), str(case), str(ht)], check=True,
                                    capture_output=True, text=True, timeout=20)
            (out / f'case_{case}_ht{ht}.txt').write_text(result.stdout)
    manifest = {'reference': 'unchanged pristine WRF module_mp_lin.F public lin_et_al',
        'source_sha256': {n: hashlib.sha256((build / n).read_bytes()).hexdigest()
                          for n in ('module_mp_radar.F', 'module_mp_lin.F', 'driver.f90')},
        'commands': commands, 'terrain_m': [0, 1500], 'cases': list(range(1, 7)),
        'init': 'compiler-defined zero; production traps off',
        'existing_tolerances': 'proofs/v060/run_lin_parity.py PREDECLARED_TOL unchanged'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['source_sha256']))


if __name__ == '__main__':
    build(Path(sys.argv[1]))
