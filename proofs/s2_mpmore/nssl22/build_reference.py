"""Staged original NSSL no-hail oracle; no product admission is implied."""
from pathlib import Path
import hashlib, json, re, subprocess, sys

if len(sys.argv)!=3:
 raise SystemExit('usage: build_reference.py SOURCE_ROOT FRESH_ARTIFACT_ROOT')
tree=Path(sys.argv[1]).resolve()
artifact=Path(sys.argv[2]).resolve()
work=artifact/'build';out=artifact/'fixtures'
if out.exists(): raise SystemExit('Use a fresh artifact root; preserve previous references')
artifact.mkdir(parents=True,exist_ok=True)
source = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F')
proof = tree/'proofs/v034/oracle/nssl2mom'
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
work.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
def run(args, **kwargs):
 return subprocess.run(list(map(str,args)),check=True,timeout=90,**kwargs)
for name,args in [('module_vars.txt',[source,1226]),('gs_vars.txt',[source,13714,12638])]:
 with (work/name).open('w') as f: run([sys.executable,proof/'list_module_vars.py',*args],stdout=f)
instr=work/'instrumented.F'
run([sys.executable,proof/'make_instrumented.py',source,instr,work/'module_vars.txt',work/'gs_vars.txt'])
driver=(proof/'nssl2mom_oracle_driver_v2.F90').read_text()
assert driver.count('nssl_hail_on=.TRUE.')==1
driver=driver.replace('nssl_hail_on=.TRUE.','nssl_hail_on=.FALSE.')
driver=driver.replace('VHL=VHL, F_VHL=.TRUE.','VHL=VHL, F_VHL=.FALSE.')
driver=driver.replace('  ! save inputs','  QHL=0.0; CHL=0.0; VHL=0.0 ! Absent Registry hail species for legacy22.\n  ! save inputs')
rows=[]
for mode in ['fp32','fp64']:
 flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off']
 if mode=='fp64': flags+=['-fdefault-real-8','-fdefault-double-8']
 for kind in ['pristine','instr']:
  b=work/(mode+'_'+kind); b.mkdir(exist_ok=True)
  (b/'module.F').write_bytes((source if kind=='pristine' else instr).read_bytes())
  (b/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes())
  (b/'driver.F90').write_text(driver)
  (b/'namelist.input').write_text('')
  for file in ['stubs.f90','module.F','driver.F90']:
   extra=['-DO1_INSTR'] if kind=='instr' and file=='driver.F90' else []
   run([fc,*flags,*extra,'-c',file],cwd=b)
  run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b)
  for case in range(1,18):
   with (b/f'case_{case}.txt').open('w') as f:
    run([b/'oracle',case,*(['dumpmv'] if kind=='instr' and case==1 else [])],cwd=b,stdout=f)
 for case in range(1,18):
  a=(work/(mode+'_pristine')/f'case_{case}.txt').read_text()
  b=(work/(mode+'_instr')/f'case_{case}.txt').read_text()
  filtered=''.join(line+'\n' for line in b.splitlines() if not re.match(r'^(S[0-9]:|MV:|MVS:|G[0-9]:)',line))
  assert a==filtered,(mode,case,'E26')
  target=out/f'{mode}_{case}.json'
  run([sys.executable,proof/'dump_to_json_v2.py',work/(mode+'_instr')/f'case_{case}.txt',target,mode])
  rows.append(dict(mode=mode,case=case,E26=True,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),work='legacy22: ipconc5,hailOFF,CCNON,densityON; 17 regimes; absent hail input zeros; REAL4/8',rows=rows),indent=2)+'\n')
print('E26 PASS34 original no-hail columns; staged references only',flush=True)
