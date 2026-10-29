"""Create isolated one-seam source deletions without editing the live branch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert a.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams'))
a.out.mkdir(parents=True,exist_ok=False)
def git(*args,input=None,env=None):
 return subprocess.check_output(['git','-C',str(a.source),*args],input=input,env=env).decode().strip()
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode!=0
parent=git('rev-parse','HEAD')
op='src/gpuwrf/runtime/operational_mode.py'
energy='src/gpuwrf/physics/noahmp/energy.py'
op_text=subprocess.check_output(['git','-C',str(a.source),'show',parent+':'+op]).decode()
energy_text=subprocess.check_output(['git','-C',str(a.source),'show',parent+':'+energy]).decode()
clock=op_text.replace('    from gpuwrf.surface_params.lsmparams import clock_controls\n','\n').replace(
    'surface_controls=clock_controls(namelist),','surface_controls=namelist.surface_controls,')
dry_line='    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)'
dry=energy_text.replace(dry_line+' if p.w2_coefficients is None or "thkdry" not in p.w2_coefficients else p.w2_coefficients["thkdry"]',dry_line)
inv_line='    satratio = smc / p.smcmax'
inverse=energy_text.replace(inv_line+' if p.w2_coefficients is None or "inv_smcmax" not in p.w2_coefficients else smc * p.w2_coefficients["inv_smcmax"]',inv_line)
plan={'clock':(op,op_text,clock),'dry_consumer':(energy,energy_text,dry),'inverse_consumer':(energy,energy_text,inverse)}
record={'parent':parent,'mutants':{}}
for name,(path,original,mutant) in plan.items():
 assert mutant!=original and mutant.count('\n')==original.count('\n')
 env={**os.environ,'GIT_INDEX_FILE':str(a.out/(name+'.index'))}
 git('read-tree',parent,env=env)
 blob=git('hash-object','-w','--stdin',input=mutant.encode())
 git('update-index','--cacheinfo','100644',blob,path,env=env)
 tree=git('write-tree',env=env)
 message=f'W2 isolated {name} deletion for numeric sensitivity\n\nCo-Authored-By: Codex <noreply@openai.com>\n'
 commit=git('commit-tree',tree,'-p',parent,input=message.encode())
 git('update-ref','refs/archive/v45-lsmparams/'+parent[:12]+'/'+name,commit,'0'*40)
 record['mutants'][name]={'commit':commit,'file':path,'sha256':hashlib.sha256(mutant.encode()).hexdigest(),
    'expected':'genuine assertion failure; no errors/skips'}
 (a.out/(name+'.index')).unlink()
(a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
