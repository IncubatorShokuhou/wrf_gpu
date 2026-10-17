"""Run genuine numerical deletion controls without editing measured source files."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/"src"))
sys.path.insert(0,str(root/"scripts/v025"))
import cpu_guard  # Establish the same pre-JAX protection pytest requires.
from gpuwrf.coupling import scan_adapters as module
import pytest

mode=sys.argv[1]
if mode in ("precip", "thermo"):
    source=subprocess.check_output(["git","show","0013f53df:src/gpuwrf/coupling/scan_adapters.py"],cwd=root,text=True)
    names={"_apply_mp_replacements"} if mode == "precip" else {"wdm6_adapter", "wdm7_adapter"}
    chosen=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(chosen)==len(names)
    code=ast.Module(body=chosen,type_ignores=[])
else:
    assert mode == "water_mask"
    node=next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="wdm6_adapter")
    text=ast.unparse(node)
    assert text.count("1.0, 2.0")==1
    code=ast.parse(text.replace("1.0, 2.0","1.0, 0.0"))
exec(compile(code,"<t-mprain-numeric-deletion-control>","exec"),module.__dict__)
selection="original_wrf_total" if mode=="precip" else "wdm_caller"
rc=pytest.main([str(root/"tests/test_non_thompson_precipitation.py"),"-q","--tb=short","-k",selection,
                "--basetemp","<USER_HOME>/wrf_gpu2_lanes/t-mprain/pytest"])
# The caller must inspect AssertionError/numeric mismatch and zero setup errors.
print(json.dumps(dict(mode=mode,pytest_rc=int(rc))))
sys.exit(int(rc))
