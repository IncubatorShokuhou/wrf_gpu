"""Bind a successful cold setup to its source, actual model process and immutable receipts."""
import hashlib,json,sys,datetime
from pathlib import Path
bundle,arm=map(Path,sys.argv[1:3]);pin,mode,input_dir=sys.argv[3:6]
assert mode in ('gpu','smoke')
assert (arm/'rc.txt').read_text().strip()=='0'
assert (arm/'source_head.txt').read_text().strip()==pin
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
if mode=='smoke' and (arm/'CPU_PLAN_ONLY.json').is_file():
    admission=json.loads((arm/'CPU_PLAN_ONLY.json').read_text());assert admission['scope']=='PLAN_ONLY'
    measured=dict(platform='cpu',observation='device preflight / CLI plan only, not model execution')
    p=dict(root_steps=None,hours=None,all_domains_finite=None,per_domain=None)
    r=dict(pin=pin,mode=mode,arm=str(arm),input_dir=input_dir,CPU_model_gate='PLAN_ONLY',GPU_native_gates='NOT_APPLICABLE',admission=admission)
else:
    measured=json.loads((arm/'device_measured.json').read_text())
    assert measured['platform']==('gpu' if mode=='gpu' else 'cpu'),measured
    proof=arm/'proof/nested_pipeline_run.json';p=json.loads(proof.read_text())
    assert p.get('all_domains_finite') is True,'nonfinite or absent whole-model evidence'
    assert (arm/'wrfout_sha.txt').is_file()
    r=dict(pin=pin,mode=mode,arm=str(arm),input_dir=input_dir,proof=str(proof),proof_sha256=sha(proof),
       sha_list=str(arm/'wrfout_sha.txt'),sha_list_sha256=sha(arm/'wrfout_sha.txt'),backend=measured,
       CPU_model_gate='ACTUAL_FIRST_ROOT' if mode=='smoke' else None,
       GPU_native_gates='PENDING_E41' if mode=='gpu' else 'NOT_APPLICABLE_CPU_SMOKE')
out=bundle/'meta/primes';out.mkdir(exist_ok=True)
with (out/(arm.name+'.json')).open('x') as f:json.dump(r,f,indent=2)
print('COLD SETUP RECEIPT',arm.name,'source',pin,'mode',mode)

now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
kind='COLD-SETUP' if mode=='gpu' else 'CPU-SEQ-SMOKE'
fields=[now,'w44-rc','Codex',kind+'-'+arm.name,'explicit cold setup then warm qualification' if mode=='gpu' else ('exact CLI plan; model NOT_APPLICABLE' if r.get('CPU_model_gate')=='PLAN_ONLY' else 'exact-call CPU launcher reaches real bounded model'),
        'source '+pin+'; model='+str(measured['platform']),
        'compile cost / whole prime wall' if mode=='gpu' else ('PLAN_ONLY; GPU gates N/A' if r.get('CPU_model_gate')=='PLAN_ONLY' else 'first-root dispatch only; GPU gates N/A'),'new private RC cache',
        json.dumps({k:p.get(k) for k in ('wall_clock_total_s','wall_clock_forecast_only_s','root_steps','hours','all_domains_finite')},sort_keys=True),
        'COMPLETE setup; release gates still PENDING' if mode=='gpu' else 'PLAN_ONLY / NOT_APPLICABLE' if r.get('CPU_model_gate')=='PLAN_ONLY' else 'CPU model phase reached; NOT GPU/science PASS',
        'artifact='+str(arm)+';cache='+str(bundle/'jax')+';autotune='+str(bundle/'autotune')+';triton='+str(bundle/'triton')+';work='+json.dumps(p.get('per_domain'),sort_keys=True)]
with Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv').open('a') as f:f.write('\t'.join(str(x).replace('\n',' ') for x in fields)+'\n')
