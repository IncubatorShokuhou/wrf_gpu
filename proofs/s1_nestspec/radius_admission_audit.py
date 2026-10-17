"""Compile unchanged WRF physics_init flags for every proposed extra MP code.

Adapted from the independently reviewed s1-nssl radius predicate extraction.
No model execution and no fitted or inferred radius values.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
WORK = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
WORK.mkdir(exist_ok=False, parents=True)
source = (WRF/'phys/module_physics_init.F').read_text()
start = source.index('   has_reqc = 0')
block = source[start:source.index('! for P3,', start)]+'   ENDIF\n'
description = (WRF/'frame/module_state_description.F').read_text()
constants = dict(re.findall(r'INTEGER, PARAMETER :: (\w+) = (\d+)', description, re.I))
needed = {word.lower() for word in re.findall(r'\b\w+\b',block)}
params = '\n'.join(f'integer,parameter::{k}={v}' for k,v in constants.items() if k.lower() in needed)
program = f'''program radius_flags
implicit none
{params}
type config_type
integer :: use_mp_re=1, ra_lw_physics=4, ra_sw_physics=4, mp_physics, nssl_2moment_on=1
end type
type(config_type)::config_flags
integer::has_reqc,has_reqi,has_reqs,mp,use_re
integer,parameter::cases(6)=[8,18,24,26,28,40]
do mp=1,6
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
compiler = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
subprocess.run([compiler,'-cpp','-ffree-line-length-none','flags.F90','-o','flags'],cwd=WORK,check=True)
result = subprocess.run([str(WORK/'flags')],cwd=WORK,check=True,capture_output=True,text=True)
rows = [list(map(int,line.split())) for line in result.stdout.splitlines()]
expected = [[mp,use_re,*([int(use_re==1 and mp!=40)]*3)]
            for mp in (8,18,24,26,28,40) for use_re in (0,1)]
assert rows == expected, rows
OUT.write_text(json.dumps({'schema':'pristine-extra-MP-radius-admission-audit-v1',
    'rows':rows,'columns':['mp_physics','use_mp_re','has_reqc','has_reqi','has_reqs'],
    'exact_init_block':block,'generated_sha256':hashlib.sha256(program.encode()).hexdigest(),
    'source_sha256':{str(p.relative_to(WRF)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (WRF/'phys/module_physics_init.F',WRF/'frame/module_state_description.F')},
    'result':'18/24/26/28 require held MP radii for default use_mp_re1/RRTMG;40 aercu0 uses defaults',
    'performance':'missing producer+held leaves+consumer; GPU launches/temps/compile costs unknown; no speed claim'
    },indent=2)+'\n')
print(result.stdout,end='')
