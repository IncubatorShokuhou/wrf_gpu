"""Host-only critic-fix scope proof; does not execute model timesteps."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root,output=map(Path,sys.argv[1:3])
baseline='89ecb0498'  # Reviewed lane rebased onto main644da7f75, before fixes.
def source(path,revision=None):
    return (subprocess.check_output(['git','show',revision+':'+path],cwd=root,text=True)
            if revision else (root/path).read_text())
def function(tree,name):
    return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)

changed=subprocess.check_output(['git','diff','--name-only',baseline,'HEAD','--','src'],cwd=root,text=True).splitlines()
expected=['src/gpuwrf/integration/nested_pipeline.py','src/gpuwrf/io/feedback_restart.py',
          'src/gpuwrf/io/nesting_options.py','src/gpuwrf/io/restart.py',
          'src/gpuwrf/io/wrfrst_netcdf.py','src/gpuwrf/runtime/restart_store.py']
assert changed==expected,changed
path=expected[0];old=ast.parse(source(path,baseline));new=ast.parse(source(path))
for name in ('_load_domains','_PerDomainWrfoutWriter'):
    assert ast.dump(function(old,name))==ast.dump(function(new,name))
driver=function(new,'_execute_nested_pipeline');removed=0;rewritten=0
for parent in ast.walk(driver):
    for name,value in ast.iter_fields(parent):
        if not isinstance(value,list):continue
        for node in list(value):
            if not isinstance(node,ast.If):continue
            if ast.unparse(node.test)!="len(names) > 1 and meta.get('nesting', {}).get('feedback')":continue
            if any(isinstance(n,ast.ImportFrom) and n.module=='gpuwrf.io.feedback_restart' for n in node.body):
                value.remove(node);removed+=1
            else:
                node.test=ast.parse("meta.get('nesting',{}).get('feedback')",mode='eval').body;rewritten+=1
assert (removed,rewritten)==(1,1)
assert ast.dump(driver)==ast.dump(function(old,'_execute_nested_pipeline'))
path='src/gpuwrf/runtime/restart_store.py'
assert ast.dump(function(ast.parse(source(path,baseline)),'nested_identity'))==ast.dump(function(ast.parse(source(path)),'nested_identity'))
path='src/gpuwrf/io/restart.py'
restorer=function(ast.parse(source(path)),'_restore_carry');restorer.body=restorer.body[2:]
assert ast.dump(restorer)==ast.dump(function(ast.parse(source(path,baseline)),'_restore_carry'))
contract=ast.parse(source('src/gpuwrf/io/feedback_restart.py'))
calls=[ast.unparse(n.func) for n in ast.walk(contract) if isinstance(n,ast.Call)]
assert not any(any(x in call for x in ('asarray','device_put','astype','jit','copy','isfinite')) for call in calls)
result=dict(verdict='PASS',baseline_rebased_before_fix=baseline,main_base='644da7f75',
    candidate=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    source_changes=changed,all_other_source_blobs_BYTE=True,
    initial_loader_and_writer_AST_identical=True,
    driver_AST_identical_except_authenticated_schema_binding_and_effective_child_guard=True,
    device_restore_AST_identical_except_preflight_validation=True,
    preexisting_identity_fixture_failure_function_AST_identical=True,
    metadata_validator_has_no_array_copy_transfer_reduction_or_compile=True,
    scientific_reuse='Accepted original CPU-WRF/oracle/default evidence carried forward; no repeated forecast/GPU work. These fixes do not modify timestep arithmetic, State/Carry leaves or kernel sources.',
    cache='NONE',backend='no JAX imported (stdlib source proof)',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
