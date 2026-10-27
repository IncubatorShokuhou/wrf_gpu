"""Seal v7 GPU launchers: fixed head cd675372 + F2 case + new CPU twin + inline component."""
import hashlib, json, os
from pathlib import Path
B=Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart'); T=B/'night_forecast_tools_v7'
S=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/WS41_snapshot'); C='cd6753726c2409ba5682b86c2d2774035b2a3215'
CPU=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/cpu_native_may04_00_lowbdy_3r/cpu_run.json')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
cpu=json.loads(CPU.read_text())
assert cpu['status']=='DONE_CLEAN' and cpu['input_identity'] and cpu['complete_times'] and cpu['rc']==0, cpu.get('status')
assert not (B/'NIGHT_FORECAST_SEAL_V7.json').exists()
v6=json.loads((B/'NIGHT_FORECAST_SEAL_V6.json').read_text())
seal={k:v for k,v in v6.items() if k not in ('files','component_recovery','component_fixed_result')}
seal.update(snapshot=str(S), commit=C,
    case='<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/native_may04_00_lowbdy',
    case_manifest=str(B/'NATIVE_CASE_MANIFEST_V2.json'),
    runner=str(T/'gpu_forecast.py'), checker=str(T/'check_forecast.py'), arm_shell=str(T/'forecast_arm.sh'),
    cpu_result=str(CPU), primary_fixed_result=str(B/'NIGHT_FORECAST_RESULT_V7.json'),
    component_runner=str(B/'night_component_tools_v3/gpu_component.py'), component_oracle=str(B/'WS11_mynn'),
    smoke_cuda_cache_receipt=str(B/'NIGHT_COMPONENT_RECOVERED_RESULT_V1.json'),
    phase_caps_s={'component_gpu':240,'E41_component_cpu':120,'primary_gpu':2400,'E41_cpu':120,'finite_hash_cpu':120},
    scope='v7: critic FIX F1-F3 head; inline MYNN component oracle (GPU gate 1) then cold 3-nest 3h on WRFDA low-bdy inputs; GA-only; run dirs on <DATA_ROOT> scratch',
    run_dirs='<DATA_ROOT>/wrf_gpu2/scratch/w44-warmstart/runs')
files=[*sorted(map(str,T.iterdir())), str(B/'night_component_tools_v1/night_groups.py'),
       str(B/'night_component_tools_v3/gpu_component.py'), str(B/'night_component_tools_v3/compare_mynn_init.py'),
       str(B/'night_component_tools_v3/launch_preflight.py'), str(B/'NATIVE_CASE_MANIFEST_V2.json'),
       str(B/'NIGHT_COMPONENT_RECOVERED_RESULT_V1.json'), str(S/'scripts/wave_cubin_gate.py'), str(CPU),
       *sorted(str(p) for p in (B/'WS11_mynn').iterdir() if p.is_file())]
seal['files']={p:sha(p) for p in files}
for p in T.iterdir(): os.chmod(p, 0o444 if p.suffix!='.sh' else 0o555)
out=B/'NIGHT_FORECAST_SEAL_V7.json'; out.write_text(json.dumps(seal,indent=1)+'\n'); os.chmod(out,0o444)
for name,tool in (('night_forecast_v7.sh','night_forecast.py'),('night_warm_cache_v7.sh','night_warm_cache.py')):
    text=f'''#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
export W44_WARMSTART_LAUNCHER={B/name}
TOOL={T/tool}
SEAL={out}
[[ $# == 0 ]] || exit 126
[[ $(sha256sum "$TOOL" | cut -d " " -f 1) == {sha(T/tool)} ]] || exit 126
[[ $(sha256sum "$SEAL" | cut -d " " -f 1) == {sha(out)} ]] || exit 126
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec python3 -u "$TOOL"
'''
    (B/name).write_text(text); os.chmod(B/name,0o555)
print(json.dumps({'seal':str(out),'seal_sha256':sha(out),'forecast':sha(B/'night_forecast_v7.sh'),'warm':sha(B/'night_warm_cache_v7.sh'),'files':len(files)}))
