"""Compile the literal pristine physics_init has_req block for mp8/18/40.

Registry radius allocation is insufficient to enable the RRTMG radius path.
For mp40, radiation_driver's separate MSKF blend can later enable it, but
only inside aercu_opt>0. The supported aercu_opt=0 point must retain defaults.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
WRF=ROOT/'data/wrf_pristine/WRF'
WORK=Path(sys.argv[1]).resolve()
WORK.mkdir(parents=True,exist_ok=True)
init=(WRF/'phys/module_physics_init.F').read_text()
start=init.index('   has_reqc = 0')
stop=init.index('! for P3,',start)
block=init[start:stop]+'   ENDIF\n'
registry=(WRF/'frame/module_state_description.F').read_text()
constants=dict(re.findall(r'INTEGER, PARAMETER :: (\w+) = (\d+)',registry,re.I))
needed={w.lower() for w in re.findall(r'\b\w+\b',block)}
params='\n'.join(f'integer,parameter::{k}={v}' for k,v in constants.items() if k.lower() in needed)
program=f'''program radius_flags
implicit none
{params}
type config_type
integer :: use_mp_re=1, ra_lw_physics=4, ra_sw_physics=4, mp_physics, nssl_2moment_on=1
end type
type(config_type)::config_flags
integer::has_reqc,has_reqi,has_reqs,mp,use_re
integer,parameter::cases(3)=[8,18,40]
do mp=1,3
do use_re=0,1
config_flags%mp_physics=cases(mp)
config_flags%use_mp_re=use_re
{block}
write(*,*) cases(mp),use_re,has_reqc,has_reqi,has_reqs
end do
end do
end program
'''
(WORK/'flags.F90').write_text(program)
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
subprocess.run([fc,'-cpp','-ffree-line-length-none','flags.F90','-o','flags'],cwd=WORK,check=True)
result=subprocess.run([str(WORK/'flags')],cwd=WORK,check=True,capture_output=True,text=True)
rows=[list(map(int,line.split())) for line in result.stdout.splitlines()]
assert rows==[[8,0,0,0,0],[8,1,1,1,1],[18,0,0,0,0],[18,1,1,1,1],[40,0,0,0,0],[40,1,0,0,0]],rows
rad=(WRF/'phys/module_radiation_driver.F').read_text()
mskf=rad[rad.index('             IF (aercu_opt.gt.0.0) THEN'):rad.index('        IF ( PRESENT ( CLDFRA_BL )')]
assert 'has_reqc = 1' in mskf and 'EFCG(I,K,J)' in mskf
receipt=dict(schema='wrf-v040-morrison-radius-activation-audit',rows=rows,columns=['mp_physics','use_mp_re','has_reqc','has_reqi','has_reqs'],
             hashes={str(p.relative_to(WRF)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (WRF/'phys/module_physics_init.F',WRF/'phys/module_radiation_driver.F',WRF/'frame/module_state_description.F')},
             exact_init_block=block,mskf_block=mskf,generated_sha256=hashlib.sha256(program.encode()).hexdigest(),
             result='mp40 aercu_opt0 does not activate radii; aercu_opt>0 MSKF path remains refused',
             command='JAX_PLATFORMS=cpu taskset -c 9 nice -n 19 timeout 60 python proofs/v040/nssl_cumulus/radius_audit.py <lane-dir>')
(ROOT/'proofs/v040/nssl_cumulus/radius_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(result.stdout,end='')
