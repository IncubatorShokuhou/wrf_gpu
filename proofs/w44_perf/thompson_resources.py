"""CPU binary inspection of the actual d02 Thompson kernel, no compile/GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import zstandard
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from gpuwrf.runtime import aot_executable as ax,aot_slim_module as asm
spec=importlib.util.spec_from_file_location('gate',ROOT/'scripts/wave_cubin_gate.py');gate=importlib.util.module_from_spec(spec);sys.modules['gate']=gate;spec.loader.exec_module(gate)
scan=gate._scanner('cubin_scan')
L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf');out=L/'THOMPSON_RESOURCES';out.mkdir(exist_ok=False)
index=json.loads((L/'ACTUAL_HLO/INDEX.json').read_text());rows=[]
for rec in index['records']:
    if rec['program']!='d02':continue
    data=zstandard.ZstdDecompressor().decompress(Path(rec['cache_entry']).read_bytes(),max_output_size=1<<33)[4:]
    native=gate._record(data,2,ax,asm);assert hashlib.sha256(native).hexdigest()==rec['native_record2_sha256']
    matched=[]
    for number,elf in enumerate(gate._elfs(native)):
        file=out/(rec['version']+'_'+str(number)+'.cubin');file.write_bytes(elf)
        try:
            tool=shutil.which('cuobjdump');assert tool
            text=subprocess.check_output([tool,'--dump-resource-usage',str(file)],text=True,timeout=30)
            resources=scan.parse_resources(text)
            for name,resource in resources.items():
                if name!='thompson_full_column':continue
                sass_text=subprocess.check_output([tool,'--dump-sass','--function',name,str(file)],text=True,timeout=30)
                funcs=scan.parse_sass(sass_text);body=funcs[name]
                from collections import Counter
                opcodes=Counter(scan._opcode(op) for op in body.values())
                matched.append(dict(resource,static_sass=len(body),opcode_counts=dict(opcodes.most_common()),
                                    elf_sha256=hashlib.sha256(elf).hexdigest()))
        finally:file.unlink(missing_ok=True)
    assert len(matched)==1
    row={'version':rec['version'],'program':'d02','cache_entry':rec['cache_entry'],'native_record2_sha256':rec['native_record2_sha256'],'kernel':matched[0]}
    rows.append(row);print(rec['version'],matched[0],flush=True)
(out/'receipt.json').write_text(json.dumps({'scope':__doc__,'records':rows},indent=2)+'\n')

