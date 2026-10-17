"""Compile the final composed CPU radiation artifact; no numerical re-solve.

Use retained coupled operands for shape/dtype/resources and the S2 fp64 audit.
Keep computational fp64 rows with their exact source frame, not a global
output-dtype assertion. Raw HLO is summarized and discarded in this process.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('completed',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();sys.path.insert(0,str(a.source/'src'))
import jax
import jax.numpy as jnp
from gpuwrf.coupling.flg_gfdl_radiation import paired_radiation
assert jax.devices()[0].platform=='cpu'
with a.completed.open('rb') as stream:carry,namelist,_start=pickle.load(stream)
model=a.source/'src/gpuwrf/physics/_gfdleta_radtn.py'
lines=set();model_ast=ast.parse(model.read_text())
routine=next(n for n in ast.walk(model_ast) if isinstance(n,ast.FunctionDef) and n.name=='_radtn')
for node in ast.walk(routine):
    if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='v'
        and isinstance(node.slice,ast.Constant) and node.slice.value in ('ddx','eex','prod')):
        lines.update(range(node.lineno,node.end_lineno+1))
fn=jax.jit(lambda state,nl,land,gfdl:paired_radiation(state,nl,jnp.float32(0.),land_state=land,gfdl=gfdl))
start=time.perf_counter();executable=fn.lower(carry.state,namelist,carry.noahmp_land,carry.gfdl_radiation).compile()
compile_s=time.perf_counter()-start
hlo=executable.as_text();memory=executable.memory_analysis()
def block(name):
    match=re.search(r'(?ms)^'+name+r'\n(.*?)(?:\n\n|\Z)',hlo)
    return '' if match is None else match[1]
files={int(i):json.loads(name) for i,name in re.findall(r'(?m)^(\d+) (".*")$',block('FileNames'))}
def records(name):
    return {int(i):{key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',data)}
        for i,data in re.findall(r'(?m)^(\d+) \{([^}]+)\}$',block(name))}
locations=records('FileLocations');frames=records('StackFrames')
source_text=model.read_text().splitlines()
def origin(row):
    frame=re.search(r'stack_frame_id=(\d+)',row)
    current=int(frame[1]) if frame else 0;seen=set()
    while current and current not in seen:
        seen.add(current);record=frames.get(current,{})
        location=locations.get(record.get('file_location_id'),{})
        filename=files.get(location.get('file_name_id'),'')
        if filename.endswith('/_gfdleta_radtn.py'):
            line=location['line']
            return dict(stack_frame_id=current,file=filename,line=line,
                source=source_text[line-1],native_DOUBLE_use=line in lines)
        current=record.get('parent_frame_id',0)
    return None
all_rows=[];computational=[];outside=[]
origins=[]
structural={'parameter','constant','copy','tuple','get-tuple-element','while','conditional','call','fusion','bitcast'}
for row in hlo.splitlines():
    if not re.search(r'\bf64\[',row):continue
    all_rows.append(row)
    match=re.search(r'=\s*f64\[[^]]*\](?:\{[^}]*\})?\s+(\w[\w-]*)\(',row)
    if not match or match[1] in structural:continue
    computational.append(row)
    source=origin(row);origins.append(dict(hlo=row,origin=source))
    if source is None or not source['native_DOUBLE_use']:outside.append(row)
report=dict(platform='cpu',GPU_executed=False,scheme=int(namelist.ra_lw_physics),compile_only=True,
    compile_s=compile_s,hlo_lines=len(hlo.splitlines()),hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
    temporary_bytes=memory.temp_size_in_bytes,argument_bytes=memory.argument_size_in_bytes,output_bytes=memory.output_size_in_bytes,
    f64_rows=len(all_rows),computational_f64=len(computational),computational_f64_outside_native_DOUBLE=len(outside),
    computational_f64_rows=computational,outside_rows=outside,native_DOUBLE_source_lines=sorted(lines),
    computational_f64_origins=origins,
    scope='final composed CPU radiation on retained real Swiss shapes; no execution/numerical repeat; no GPU/readiness/forecast-speed claim')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.endswith('_rows') and k!='native_DOUBLE_source_lines'},indent=2))
