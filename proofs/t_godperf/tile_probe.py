"""One official-lock GPU component bakeoff, preserving all column arithmetic."""
import ast
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

source, inputs, out = map(Path, sys.argv[1:4])
assert os.environ['GPUWRF_GPU_LOCK_HELD'] == '1'
assert subprocess.run(['git','-C',str(source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode
sys.path.insert(0, str(source/'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling import goddard_radiation as gr
assert jax.devices()[0].platform == 'gpu'
out.mkdir(parents=True,exist_ok=False)
os.chmod(out,0o700)
rec=dict(backend='gpu', device=str(jax.devices()[0]), source_commit=subprocess.check_output(
    ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
    scope='real c45 initial columns night + same-state noon astronomy; component, not forecast/S1/fidelity',
    affinity=sorted(os.sched_getaffinity(0)), cache=str(out.parent/'cache'), rows=[])
def save():
    rec['peak_rss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'receipt.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
    print(json.dumps(rec['rows'][-1] if rec['rows'] else {'phase':'init'}),flush=True)
save()
data={}
capture=json.loads((inputs/'receipt.json').read_text())
for regime in ('night','day'):
    path=inputs/(regime+'.npz')
    seal=next(r for r in capture['captures'] if Path(r['file']).name==path.name)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['sha256']
    with np.load(path) as f:host={k:f[k] for k in f.files}
    julday=host.pop('julday');solcon=host.pop('solcon')
    data[regime]=(jax.tree.map(jnp.asarray,host),jnp.asarray(julday),jnp.asarray(solcon))
jax.block_until_ready(data)
base_source=inspect.getsource(gr.solve_goddard_columns.__wrapped__)
rec['solver_source_sha256']=hashlib.sha256(base_source.encode()).hexdigest()
compiled={}
for tile in (128,1024,2048,4096):
    if tile==128:
        fn=gr.solve_goddard_columns
    else:
        tree=ast.parse(base_source)
        class TileOnly(ast.NodeTransformer):
            changes=0
            def visit_Constant(self,node):
                if type(node.value) is int and node.value==128:
                    self.changes+=1
                    return ast.copy_location(ast.Constant(tile),node)
                return node
        transformer=TileOnly();tree=transformer.visit(tree)
        assert transformer.changes==2, transformer.changes
        ns=dict(gr.__dict__)
        exec(compile(ast.fix_missing_locations(tree),str(source/'src/gpuwrf/coupling/goddard_radiation.py'),'exec'),ns)
        fn=ns['solve_goddard_columns']
    started=time.monotonic()
    executable=fn.lower(*data['night']).compile()
    path=out/f'tile{tile}.xlaexec'
    path.write_bytes(executable.runtime_executable().serialize())
    mem=executable.memory_analysis()
    rec['rows'].append(dict(tile=tile,phase='compiled',compile_s=time.monotonic()-started,
        executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        memory={name:getattr(mem,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}))
    save();compiled[tile]=executable
cudart=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
baseline={}
# Warm every executable/regime before the profiler window; retain exact host
# outputs outside timing. Save timestamps for independent SQLite range joins.
for tile,exe in compiled.items():
    for regime,args in data.items():
        started=time.monotonic();values=exe(*args);jax.block_until_ready(values)
        wall=time.monotonic()-started
        arrays=[np.asarray(v) for v in values]
        assert all(np.isfinite(v).all() for v in arrays)
        if tile==128:baseline[regime]=arrays
        equal=[a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes() for a,b in zip(arrays,baseline[regime])]
        rec['rows'].append(dict(tile=tile,regime=regime,phase='warmup',wall_s=wall,
            byte_equal=equal,all_byte_equal=all(equal), max_abs=[float(np.max(np.abs(a-b))) for a,b in zip(arrays,baseline[regime])]))
        save()
assert cudart.cudaProfilerStart()==0
for tile,exe in compiled.items():
    for regime,args in data.items():
        # CLOCK_MONOTONIC shares the Nsight Linux timestamp basis; CUDA events
        # and HtoD/DtoH outside these execution-only intervals are excluded.
        start_ns=time.monotonic_ns()
        values=exe(*args);jax.block_until_ready(values)
        end_ns=time.monotonic_ns()
        rec['rows'].append(dict(tile=tile,regime=regime,phase='profile',
            start_ns=start_ns,end_ns=end_ns,wall_s=(end_ns-start_ns)*1e-9))
        save()
assert cudart.cudaProfilerStop()==0
rec['all_byte_equal']=all(r['all_byte_equal'] for r in rec['rows'] if r['phase']=='warmup')
rec['device_memory_stats']={k:int(v) for k,v in (jax.devices()[0].memory_stats() or {}).items()}
save()
os._exit(0)
