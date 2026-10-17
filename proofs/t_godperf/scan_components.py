"""E41 scan of every actual GPU-compiled tile probe executable, on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

source, arm = map(Path,sys.argv[1:3])
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path[:0]=[str(source/'src'),str(source/'scripts')]
import wave_cubin_gate as gate
out=arm/'e41'
out.mkdir(exist_ok=False)
rows=[]
for path in sorted(arm.glob('tile*.xlaexec')):
    row=gate.scan_executable(gate.Executable(path,'aot',path.stem),out,out/'scan.sqlite')
    rows.append(row)
assert len(rows)==4
rec=dict(scope='actual component CUDA executable E41; not whole-model E41',entries=rows,
    pass_stack=all(not e['stack_over256'] for r in rows for e in r['elfs']),
    pass_nonleaf=all(not e['nonleaf_over1k'] and not e['callgraph_unknown'] for r in rows for e in r['elfs']),
    pass_kinput=all(r['kinput_transpose_status'].startswith('MEASURED_') and not r['kinput_transpose_over500'] for r in rows))
(out/'gate.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({k:v for k,v in rec.items() if k!='entries'},indent=2))
