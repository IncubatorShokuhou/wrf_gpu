"""Export one immutable tool set and its manifest variants (production, CUDA mini, CPU-only coupled rehearsal)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--mini-case',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    pin=subprocess.check_output(['git','-C',str(args.model),'rev-parse','HEAD'],text=True).strip()
    tree=subprocess.check_output(['git','-C',str(args.model),'rev-parse','HEAD:src'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(args.model),'status','--porcelain','--untracked-files=no'],text=True).strip()
    base=Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget')
    old=json.loads((base/'day_tools_v5_36d10cacc5/manifest.json').read_text())
    args.out.mkdir(exist_ok=False)
    files=[*old['files'],'night_common_v6.py','with_memory.sh']
    for variant in ('production','cuda_mini','cpu_rehearsal_coupled'):
        target=args.out/variant;target.mkdir()
        for name in files:
            data=subprocess.check_output(['git','show',args.commit+':proofs/w44_budget/'+name])
            (target/name).write_bytes(data)
        manifest=dict(old,schema='wrf_gpu2.w44_budget.launch_manifest.v7',grid_profile=variant,
                      model=str(args.model),model_pin=pin,source_tree=tree,
                      cache_root='<DATA_ROOT>/wrf_gpu2/scratch/w44-budget/night_design_b_cache_'+pin[:10],
                      tool_commit=args.commit,files={name:sha(target/name) for name in files},
                      ram_policy='own systemd scope MemoryMax=12G MemorySwapMax=0, no shared FIFO',
                      data_free_floor_gib=365,coupled=True,expected_probe_verdict='FIRST_STEP_AND_REPLAY_BYTE',
                      component_cases=[[1,'day'],[1,'night'],[0,'day'],[0,'night']],
                      # One GPU process cold-compiles 8 components + OFF/ON/EDMF0 three-nest programs: a cold
                      # WN3 3-nest GPU process alone peaked at 13.6 GB host VmHWM [M FINDINGS D26] > 12 GiB.
                      # Manager 03:40Z: 32G for the RAM-frozen window. [I] host RSS of the one GPU process: D26 13.6 GB
                      # (one cold full WN3 3-nest set) + <= ~5 GB per further program set (P70c CPU: 16.5 -> 21.6 GB
                      # for the ON set; GPU executables are mostly device-resident) x 2 sets + nsys ~1-2 GB = ~18-25 GB.
                      gpu_memory_max='32G',
                      # QA A1: post = E41 600 + 4 oracles x 150 + 2 nsys exports x 150 s, each bounded by the post deadline.
                      cpu_cap_s=1500)
        if variant=='production':
            # Post admitted with >= 720 s left and granted min(1500, remaining) (QA 01:21Z); cap_s stays 4500.
            manifest['post_min_s']=720
        if variant!='production':
            manifest.update(case=str(args.mini_case),input_receipt=str(args.mini_case/'input_receipt.json'),
                component_columns=1,duration_args=['--minutes','1.8'],max_dom=3,expected_history_files=6,
                cache_root=str(base/('cuda_mini_cache_'+pin[:10])),spill_root=str(base/'cuda_mini_runs'),
                cap_s=720,gpu_cap_s=300,cpu_cap_s=360,gpu_memory_max='12G')
        if variant=='cuda_mini':
            # <=5 min GPU: one component case (EDMF on, day, OFF+ON programs) through the SAME builders,
            # product CUDA env, lock, RAM scopes, native exports, E41 and original-term checker. A cold
            # 3-nest program compile alone exceeds 300 s ([M] ON fused > 300 s, night v1), so no replay;
            # P68 CPU compile of two cases was 262 s and GPU/CPU compile ratio ~0.9 (d01 170/189 s).
            manifest.update(component_cases=[[1,'day']],coupled=False,
                            expected_probe_verdict='COMPONENTS_BYTE')
        if variant=='cpu_rehearsal_coupled':
            # CPU device-substitution rehearsal of the coupled replay code on the crop ONLY; gpu_cap_s=1
            # makes an accidental GPU start useless. Never posted, never queued.
            # CPU compile of the fused d02/d03 program hit the 12 GiB memcg OOM (anon 12.5 GB, P65/P70).
            # 16 GiB OOM-killed right after the fused CPU compile (P70b, anon 16.7 GB) -> 24 GiB (manager approval needed to run).
            manifest.update(component_cases=[],cap_s=60,gpu_cap_s=1,cpu_cap_s=30,gpu_memory_max='32G',
                            cache_root=str(base/('cpu_rehearsal_cache_'+pin[:10])))
        post_min=manifest.get('post_min_s',min(720,manifest['cpu_cap_s']))
        assert post_min<=manifest['cpu_cap_s'] and manifest['gpu_cap_s']<=manifest['cap_s'],(variant,post_min)
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        checks=[(target/'manifest.json',sha(target/'manifest.json'))]
        checks += [(target/name,sha(target/name)) for name in files]
        common=base/'night_tools_5367b7fda6/night_supervisor.py'
        checks.append((common,sha(common)))
        script='#!/usr/bin/env bash\n# Immutable; manager sequencer only.\nset -euo pipefail\n'
        script+='export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1\n'
        script+='[[ $# == 0 ]] || exit 126\n'
        script+='check() { [[ $(sha256sum "$1" | awk \'{print $1}\') == "$2" ]] || exit 126; }\n'
        script+=''.join('check '+str(path)+' '+digest+'\n' for path,digest in checks)
        script+='exec python3 -u '+str(target/'night_supervisor_design_b.py')+'\n'
        launcher=args.out/(variant+'.sh');launcher.write_text(script);launcher.chmod(0o555)
        for path in target.iterdir():path.chmod(0o444)
        print(variant,launcher,sha(launcher))


if __name__=='__main__':main()
