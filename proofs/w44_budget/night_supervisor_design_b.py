"""Versioned design-B protocol on the tested signal/subreaper supervisor."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
MANIFEST=json.loads((HERE/'manifest.json').read_text())
COMMON=HERE/'night_common_v6.py'
assert hashlib.sha256(COMMON.read_bytes()).hexdigest()==MANIFEST['files']['night_common_v6.py']
spec=importlib.util.spec_from_file_location('budget_common',COMMON)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
M.CAP,M.GPU_CAP,M.E41_CAP=(MANIFEST.get(k,d) for k,d in
    [('cap_s',4500),('gpu_cap_s',3600),('cpu_cap_s',720)])
M.POST_MIN=MANIFEST.get('post_min_s',min(720,M.E41_CAP))  # QA 01:21Z: shorten, never skip, the post
M.SNAP=Path(MANIFEST['model']);M.PIN=MANIFEST['model_pin']
M.EXPECTED_VERDICT=MANIFEST.get('expected_probe_verdict','FIRST_STEP_AND_REPLAY_BYTE')
M.RUNNER_SHA=hashlib.sha256((M.SNAP/'proofs/w44_budget/coupled_probe.py').read_bytes()).hexdigest()
# QA A5: never inherit flag/cheap-key env from the sequencer shell; every child sets its own explicitly.
SCRUBBED=sorted(k for k in M.CPU_ENV if k.startswith(('GPUWRF_','XLA_','JAX_')))
for k in SCRUBBED:del M.CPU_ENV[k]
M.CPU_ENV.update(JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0')
M.CPU_ENV.update(PYTHONPATH=str(M.SNAP/'src'),GPUWRF_AOT_KEEP_TRACE_METADATA='1',BUDGET_MODEL_ROOT=str(M.SNAP))
DRIVER=HERE/'night_probe_design_b.py'
M.GPU_BODY=r'''
set -euo pipefail
spill=SPILL_ROOT/$(basename -- "$RUN")
cache=CACHE_ROOT
mkdir -p "$cache/jax" "$cache/triton" "$cache/cuda" "$spill/tmp" "$spill/work"
ln -s "$cache" "$RUN/cache"
ln -s "$spill/work" "$RUN/work"
ln -s "$spill/probe" "$RUN/probe"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RUN/gpu_start.txt"
exec env JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_MYNN_EDMF=1 GPUWRF_NESTED_AOT=1 \
 GPUWRF_JAX_CACHE=1 GPUWRF_AOT_KEEP_TRACE_METADATA=1 \
 GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
 GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/jax/xla_gpu_per_fusion_autotune_cache_dir" \
 TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda" TMPDIR="$spill/tmp" \
 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=.22 \
 GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF PYTHONPATH="$SNAP/src" \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
 --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --force-overwrite=true \
 -o "$spill/work/budget_trace" python -u DRIVER --out "$spill/probe"
'''.replace('DRIVER',str(DRIVER)).replace('CACHE_ROOT',MANIFEST['cache_root']).replace(
    'SPILL_ROOT',MANIFEST.get('spill_root','<DATA_ROOT>/wrf_gpu2/scratch/w44-budget'))
write=M.write_json
def write_record(path,value):
    if path.name=='night_receipt.json':
        value=dict(value,protocol_version=4,model_source=MANIFEST['model_pin'],manifest=MANIFEST,inherited_env_scrubbed=SCRUBBED,
            scientific_gate='physical carry BYTE, actual native E41, original per-term gates; accounting UNKNOWN never equals measured')
    return write(path,value)
M.write_json=write_record
run_group=M.run_group
def grouped(command,log,limit,*,env=None):
    if Path(log).name=='gpu.log':
        # Own hard-capped scope: never fail admission behind another lane's FIFO.
        # The common supervisor owns every group/descendant across the scope.
        command=['bash',str(HERE/'with_memory.sh'),'--max',MANIFEST.get('gpu_memory_max','12G'),'--',*command]
    if Path(log).name=='e41.log':
        slot=next(i for i,v in enumerate(command) if Path(v).name=='with_ram_slot.sh')
        end=command.index('--',slot)+1
        command=[*command[:slot],'bash',str(HERE/'with_memory.sh'),'--',*command[end:]]
        command=[*command[:command.index(sys.executable)+1],str(HERE/'budget_post_cpu.py'),
                 str(Path(log).parent/'cache/jax'),str(Path(log).parent/'e41'),
                 '--run-proof',str(Path(log).parent/'probe/compiled_program_inventory.json')]
        # The post bounds its inner children by the ACTUAL outer limit (QA A1), also under rehearsal caps.
        env=dict(env if env is not None else M.CPU_ENV,BUDGET_POST_CAP_S=f'{float(limit):.0f}')
    return run_group(command,log,limit,env=env)
M.run_group=grouped


if __name__=='__main__':raise SystemExit(M.main())
