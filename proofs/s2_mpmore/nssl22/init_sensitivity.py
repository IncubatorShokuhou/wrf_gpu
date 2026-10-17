"""Compiler-init sensitivity without changing pristine source bodies."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
tree=Path(sys.argv[1]);lane=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore')
b=lane/'nssl22_init_build';b.mkdir(exist_ok=False)
proof=tree/'proofs/v034/oracle/nssl2mom'
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F')
driver=(proof/'nssl2mom_oracle_driver_v2.F90').read_text().replace('nssl_hail_on=.TRUE.','nssl_hail_on=.FALSE.').replace('VHL=VHL, F_VHL=.TRUE.','VHL=VHL, F_VHL=.FALSE.').replace('  ! save inputs','  QHL=0.0; CHL=0.0; VHL=0.0\n  ! save inputs')
(b/'module.F').write_bytes(source.read_bytes());(b/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes());(b/'driver.F90').write_text(driver);(b/'namelist.input').write_text('')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=inf','-finit-integer=0','-finit-logical=false','-ffp-contract=off']
run=lambda args,**kw:subprocess.run(list(map(str,args)),check=True,timeout=120,cwd=b,**kw)
for p in ('stubs.f90','module.F','driver.F90'):run([fc,*flags,'-c',p])
run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'])
rows=[]
for case in range(1,18):
 with (b/f'case{case}.txt').open('w') as f:run([b/'oracle',case],stdout=f)
 subprocess.run([sys.executable,proof/'dump_to_json_v2.py',b/f'case{case}.txt',b/f'case{case}.json','fp32'],check=True,timeout=30)
 alt=json.loads((b/f'case{case}.json').read_text());base=json.loads((tree/f'proofs/s2_mpmore/nssl22/fixtures/fp32_{case}.json').read_text())
 changes=[sec+'.'+k for sec in ('columns','scalars') for k,v in alt[sec].items() if v!=base[sec][k]]
 rows.append(dict(case=case,changed_fields=changes))
result=dict(scope='Original22 REAL4 localcompilerinit INF vs ZERO; no pristine body edits; outputcolumns/scalars only',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),flags=flags,rows=rows,changed_cases=sum(bool(r['changed_fields']) for r in rows))
(lane/'MPM22S-init-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
print('INIT sensitivity changedcases',result['changed_cases'],flush=True)
