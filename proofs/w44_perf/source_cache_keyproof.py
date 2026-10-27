"""Exact-source/value proof carrying existing PROD/WN3 AOT-key validation."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
src=L/'snap_source_cache'
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']=='0'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0,str(src/'src'))
from gpuwrf.runtime import aot_cheap_key as ck
sha=lambda b:hashlib.sha256(b).hexdigest()
baseline='61b6bcb6f9'
prior=subprocess.check_output(['git','show',baseline+':src/gpuwrf/runtime/aot_cheap_key.py'])
current=Path(ck.__file__).read_bytes()
allowed={'_import_nodes','_source_import_cache_dir','_import_cache_payload','_source_import_specs','_gpuwrf_imports_in_file'}

def normalize(data):
    module=ast.parse(data)
    body=[]
    for node in module.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in allowed: continue
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SOURCE_IMPORT_CACHE_SCHEMA' for t in node.targets):continue
        if isinstance(node,ast.Import):
            node=copy.deepcopy(node);node.names=[n for n in node.names if n.name not in ('json','tempfile')]
            if not node.names:continue
        body.append(node)
    return ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)

assert normalize(prior)==normalize(current),'unexpected key algorithm change'
old_defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(prior).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new_defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(current).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
same=[name for name in old_defs if name not in allowed and old_defs[name]==new_defs[name]]
assert len(same)==len(set(old_defs)-allowed)
ck._source_import_cache_dir=lambda:L/'SOURCE_CACHE_BENCH/meta'
candidate_root=src/'src/gpuwrf'
files=ck._trace_reachable_source_files(candidate_root)
sealed_root=Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src/src/gpuwrf')
manifest={}
for path in files:
    rel=path.relative_to(candidate_root).as_posix();a=path.read_bytes();b=(sealed_root/rel).read_bytes()
    assert a==b,rel;manifest[rel]=sha(a)
assert 'runtime/aot_cheap_key.py' not in manifest
content=ck._source_tree_content_digest(candidate_root,files)
wn3=json.loads((L/'WN3_SDS_KEYS/receipt.json').read_text())
assert content==wn3['trace_source_fingerprints'][0]
full=ck.source_fingerprint_hash()
timing=json.loads((L/'SOURCE_CACHE_BENCH/receipt.json').read_text())
assert all(r['fingerprint']==full for r in timing['records'])
step=next(row for row in wn3['keys'] if row['label']=='all_step_programs')
assert step['before']==step['after']
assert all(row['components']['source_fingerprint']==full for row in step['after'].values())
key_count=sum(1 for row in wn3['keys'] if row['label'].startswith('init:'))+len(step['after'])
out=dict(candidate_commit=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),
         candidate_module_sha256=sha(current),baseline_key_algorithm_commit=baseline,
         key_algorithm_ast_unchanged_except_import_discovery=True,unchanged_definitions=same,
         source_closure_files=len(manifest),source_closure_manifest_sha256=sha(json.dumps(manifest,sort_keys=True).encode()),
         source_closure_all_files_byte_equal_sealed_v043=True,source_content_digest=content,
         source_fingerprint=full,source_fingerprint_byte_equal_all_10_fresh_processes=True,
         prior_real_keys_receipt_sha256=sha((L/'WN3_SDS_KEYS/receipt.json').read_bytes()),
         prior_real_keys_count=key_count,source_closure_manifest=manifest,
         conclusion='Exact determinant/source proof: existing zero-SDS PROD/WN3 keys remain byte-identical; prior native/numerical validation carries. No new real init/key census claimed. Direct cold/cached/legacy AOT key equality is also exercised by the focused tests.')
(L/'SOURCE_CACHE_KEY_PROOF.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('source_closure_manifest','unchanged_definitions')},indent=2))

