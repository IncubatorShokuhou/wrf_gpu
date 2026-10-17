"""Fresh bounded pair; CPU-only supervisor, official GPU lock and one RAM slot."""
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round4_qhmass')
MAIN=Path('<USER_HOME>/src/wrf_gpu2')
assert os.environ['JAX_PLATFORMS']=='cpu'
assert os.sched_getaffinity(0)=={9}
assert not Path('/tmp/wrf_gpu2_quiet').exists()
names=sys.argv[1:]
plan=json.loads((ROOT/'RUN_PLAN.json').read_text())
assert names in plan['groups']
assert datetime.now(timezone.utc)<datetime(2026,10,15,17,30,tzinfo=timezone.utc)
assert all(not (ROOT/n/'GPU_DONE.json').exists() for n in names)
source=Path(plan['source'])
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==plan['source_rev']
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
tag='_'.join(names)
holder=Path('/tmp/wrf_gpu2_gpu.lock.holder').read_text().strip()
for name in names:
    with (ROOT/name/'gpu_prior.txt').open('w') as f:
        subprocess.run(['rg','-i',name+'|qh.mass|hail.mass',*[str(MAIN/'.agent/kernel'/x) for x in ('FINDINGS.md','EXPERIMENTS.tsv','INBOX.md')]],stdout=f)
cmd=[str(MAIN/'scripts/with_ram_slot.sh'),'--label','v-run','--max','12G','--timeout','1800','--','bash','-c',
     'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"','v-run-pair',
     'timeout','--signal=TERM','--kill-after=30','5460',str(MAIN/'scripts/with_gpu_lock.sh'),'--label','v-run','--timeout','3600','--',
     'timeout','--signal=TERM','--kill-after=15','1800',sys.executable,str(ROOT/'tools/batch.py'),*names]
(ROOT/f'{tag}_COMMAND.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':cmd,'source_rev':plan['source_rev'],'observed_predecessor':holder,'authority':plan['authority']},indent=2)+'\n')
with (ROOT/f'{tag}_pair.log').open('w') as f:
    rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
(ROOT/f'{tag}_WRAPPER_DONE.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'rc':rc,'scope':'wrapper only; science requires actual process/frame/E41/D6 evidence'})+'\n')
print(tag,'WRAPPER_DONE',rc,flush=True)
sys.exit(rc)
