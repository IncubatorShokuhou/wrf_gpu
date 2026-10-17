"""One fresh, bounded official-lock pair after explicit higher-priority release."""
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp28_followup')
MAIN=Path('<USER_HOME>/src/wrf_gpu2')
p=json.loads((ROOT/'RUN_PLAN.json').read_text())
assert os.environ['JAX_PLATFORMS']=='cpu'
assert os.sched_getaffinity(0)=={9}
assert not Path('/tmp/wrf_gpu2_quiet').exists()
release=json.loads((ROOT/'ORDER_RELEASE.json').read_text())
assert release['mode'] in ('PREDECESSOR_DONE', 'QUEUE_BEHIND_ACTIVE_PREDECESSOR')
if release['mode']=='QUEUE_BEHIND_ACTIVE_PREDECESSOR':
    current=Path('/tmp/wrf_gpu2_gpu.lock.holder').read_text().strip()
    assert current.startswith('holder=v-prep ') and current==release['active_holder'], current
nightly=json.loads(Path('<DATA_ROOT>/alisios/state/nightly_9nest/LAST_RUN.json').read_text())
assert (nightly['issue'],nightly['verdict'])==('20261014_12z','DONE')
for c in p['configs']:
    assert not (ROOT/c['config']/'GPU_DONE.json').exists()
    now=datetime.now(timezone.utc).isoformat()
    with (MAIN/'.agent/kernel/EXPERIMENTS.tsv').open('a') as f:
        f.write(f"{now}\tv-run\tGPT Sol\tVR-FU28-GPU-{c['config']}-pre\tSelected MP28 emissions model changed; full affected qualification\timmutable{p['source_rev']}>=a050;originalCPUrefs+sameinputs\t25frames/fullCPUinventory/frozenD6/E41/F2\tround2oldmodel\tPRE-REGISTER\tRUN\tartifact={ROOT/c['config']}/gpu_run.json;cache={c['runtime']}/cache(jax,autotune,triton,cuda,xdg);work=Swiss42x42x44dt18hours24output60minFAST-CLI/CENSUS1/SLIM0/RE1,officiallock v-run+12GSwap0RAMslot,arm1200s/child1170s,coldconcurrentNOTS1\n")
cmd=[str(MAIN/'scripts/with_ram_slot.sh'),'--label','v-run','--max','12G','--timeout','1800','--','bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"','v-run-pair','timeout','--signal=TERM','--kill-after=30','4860',str(MAIN/'scripts/with_gpu_lock.sh'),'--label','v-run','--timeout','3600','--','timeout','--signal=TERM','--kill-after=15','1200',sys.executable,str(ROOT/'tools/batch.py'),'mp28']
(ROOT/'PAIR_COMMAND.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':cmd,'source_rev':p['source_rev'],'order_release':release},indent=2)+'\n')
with (ROOT/'pair.log').open('w') as log:
    rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
(ROOT/'PAIR_WRAPPER_DONE.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'rc':rc,'scope':'wrapper result; not a scientific verdict; actual child receipts required'})+'\n')
print('PAIR_WRAPPER_DONE',rc,flush=True)
sys.exit(rc)
