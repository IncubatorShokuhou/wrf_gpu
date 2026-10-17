"""Stage pristine NSSL source + caller fixture, core 9 only, no JAX/GPU.
Usage: python build.py <lane-artifact-dir>
The instrumented arm only prints pre/post calcnfromcuten; final bytes must
match the UNMODIFIED source arm. Assets include source hashes and tolerances.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(sys.argv[1]).resolve()
OUT = ROOT / 'proofs/v040/nssl_cumulus/fixtures'
SOURCE = ROOT / 'data/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F'
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
BASE = ROOT / 'proofs/v034/oracle/nssl2mom'
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
driver = (BASE / 'nssl2mom_oracle_driver_v2.F90').read_text()
driver = driver.replace('INTEGER :: thermo_id, hail_seed, cn_mode', 'INTEGER :: thermo_id, hail_seed, cn_mode, cu_used')
driver = driver.replace('cn_mode   = 0', 'cn_mode   = 0\n  cu_used = 1')
driver = driver.replace('  END SELECT', '''  CASE (18, 22, 23, 24)
    thermo_id = 4; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (19)
    thermo_id = 3; itimestep = 2; DT = 18.0; cn_mode = 1
  CASE (20, 21)
    thermo_id = 4; itimestep = 1; DT = 18.0; cn_mode = 0
  END SELECT''', 1)
# A mass-only start makes CN reset and calcnfromq observable at driver level.
driver = driver.replace('  ! save inputs', '''  IF (case_id == 20 .or. case_id == 21) THEN
    CCW=0.; CRW=0.; CCI=0.; CSW=0.; CHW=0.; CHL=0.; VHW=0.; VHL=0.
    CN=1.e8
  END IF
  ! save inputs''')
driver = driver.replace('  RAINNC=0.;', '''  IF (case_id == 18 .or. case_id == 19 .or. case_id == 20) THEN
    QRCUTEN = 1.e-6; QSCUTEN = 2.e-6
    QCCUTEN = 3.e-6; QICUTEN = 4.e-6
  ELSE IF (case_id == 21 .or. case_id == 23 .or. case_id == 24) THEN
    QCCUTEN = 3.e-6; QICUTEN = 4.e-6
    IF (case_id == 24) cu_used = 0
  ELSE IF (case_id == 22) THEN
    DO k=1,KX
      SELECT CASE (MOD(k,4))
      CASE(0)
        QRCUTEN(1,k,1)=1.e-6; QSCUTEN(1,k,1)=2.e-6
      CASE(1)
        QRCUTEN(1,k,1)=-1.e-6; QSCUTEN(1,k,1)=-2.e-6
      CASE(2)
        QRCUTEN(1,k,1)=1.e-16; QSCUTEN(1,k,1)=1.e-16
      END SELECT
    END DO
  END IF
  RAINNC=0.;''', 1)
driver = driver.replace('CU_USED=0,', 'CU_USED=cu_used,')
driver = driver.replace("  WRITE(*,'(A,I0)') 'CASE=', case_id", "  WRITE(*,'(A,I0)') 'CU_USED=', cu_used\n  WRITE(*,'(A,I0)') 'CASE=', case_id")
driver = driver.replace('  ! outputs (post-NSSL)', "  CALL dump_col3('QRCUTEN',QRCUTEN)\n  CALL dump_col3('QSCUTEN',QSCUTEN)\n  CALL dump_col3('QCCUTEN',QCCUTEN)\n  CALL dump_col3('QICUTEN',QICUTEN)\n  ! outputs (post-NSSL)")
module = SOURCE.read_text()
def dump(stage):
    return f'''\n         DO il=1,na
          DO kz=kts,kte
           WRITE(*,'(A,I0,A,I0,A,ES23.15)') '{stage}:AN:',il,':',kz,'=',an(its,1,kz,il)
          END DO
         END DO\n'''
marker = '         call calcnfromcuten(nx,ny,nz,ancuten,an,na,nor,nor,dn1)'
assert module.count(marker) == 1
instrumented = module.replace(marker, dump('PRECU') + marker + dump('POSTCU'))
manifest = dict(schema='wrf-v040-nssl-cumulus', source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                units=dict(QRCUTEN='kg/kg/s', QSCUTEN='kg/kg/s', number_stage='#/m3'),
                tolerances=dict(helper_fp32_rtol=2e-6, helper_fp64_rtol=1e-12, driver_fp64_rtol=1e-10,
                                driver_fp64_operand_atol_factor=1e-13, driver_fp32_dual_rtol=2e-3,
                                driver_fp32_column_atol_factor=1e-6), cases=list(range(18,25)),
                command='JAX_PLATFORMS=cpu taskset -c 9 nice -n 19 timeout 600 python proofs/v040/nssl_cumulus/build.py <lane-artifact-dir>',
                flags='-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp', compiler=FC)
for mode in ('fp32', 'fp64'):
    promotion = [] if mode == 'fp32' else ['-fdefault-real-8', '-fdefault-double-8']
    records = {}
    for arm, text in [('pristine', module), ('printonly', instrumented)]:
        b = WORK / f'{mode}_{arm}'
        b.mkdir(exist_ok=True)
        (b / 'module.F').write_text(text)
        (b / 'driver.F90').write_text(driver)
        (b / 'stubs.f90').write_bytes((BASE / 'nssl2mom_stub_modules.f90').read_bytes())
        flags = ['-O2','-ffree-form','-ffree-line-length-none','-ffpe-summary=none','-cpp', *promotion]
        subprocess.run([FC, *flags, '-c','stubs.f90','module.F','driver.F90'], cwd=b, check=True)
        subprocess.run([FC, *flags,'-o','oracle','stubs.o','module.o','driver.o'], cwd=b, check=True)
        (b / 'namelist.input').touch()
        for case in manifest['cases']:
            run = subprocess.run([str(b/'oracle'),str(case)], cwd=b, check=True, capture_output=True, text=True)
            (b/f'case_{case}.txt').write_text(run.stdout)
            records[(arm, case)] = run.stdout
    for case in manifest['cases']:
        raw = records[('printonly',case)]
        clean = ''.join(line for line in raw.splitlines(keepends=True) if not line.startswith(('PRECU:', 'POSTCU:')))
        assert clean == records[('pristine',case)], (mode,case,'E26')
        scalars, columns, stages = {}, {}, {}
        for line in raw.splitlines():
            match = re.match(r'^(PRECU|POSTCU):AN:(\d+):(\d+)=\s*(.*)',line)
            if match:
                st, il, k, v = match.groups()
                stages.setdefault(st,{}).setdefault(int(il),{})[int(k)] = float(v)
                continue
            match = re.match(r'^([A-Z0-9_]+)\[(\d+)\]=\s*(.*)',line)
            if match:
                name,k,v = match.groups()
                columns.setdefault(name,{})[int(k)] = float(v)
                continue
            match = re.match(r'^([A-Z0-9_]+)=\s*(.*)',line)
            if match:
                name,v = match.groups()
                scalars[name] = float(v)
        columns = {n:[v[k] for k in sorted(v)] for n,v in columns.items()}
        stages = {n:[[v[k] for k in sorted(v)] for il,v in sorted(a.items())] for n,a in stages.items()}
        data = dict(mode=mode,scalars=scalars,columns=columns,stages=stages, source_sha256=manifest['source_sha256'], E26='BYTE')
        dest = OUT / f'{mode}_{case}.json'
        dest.write_text(json.dumps(data,separators=(',',':'))+'\n')
        manifest.setdefault('fixtures',{})[dest.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(mode,case,'E26 BYTE',flush=True)
manifest['driver_sha256'] = hashlib.sha256(driver.encode()).hexdigest()
manifest['printonly_sha256'] = hashlib.sha256(instrumented.encode()).hexdigest()
(OUT / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
