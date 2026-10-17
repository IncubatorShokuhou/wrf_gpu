"""Delete actual SF4 forwarding assignments in-process; retain the real tests."""
from pathlib import Path
import ast
import inspect
import sys

root, mode, output = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
sys.path[:0] = [str(root/'scripts/v025'), str(root/'src'), str(root/'tests')]
import cpu_guard  # Install before any JAX import, as the real test harness does.
import pytest
from gpuwrf.coupling import qnse_adapter as adapter
from gpuwrf.runtime import operational_mode as op

if mode == 'history':
    module, name = op, '_physics_step_forcing'
    needle = 'u10e=qnse_surface.u10e, v10e=qnse_surface.v10e'
    replacement = '**{}'
    node = 'test_qnse_single_surface_solve_populates_output_cache'
else:
    assert mode == 'producer'
    module, name = adapter, 'qnse_surface_step'
    needle = "None,surface['u10e'],surface['v10e'])"
    replacement = "None,surface['u10'],surface['v10'])"
    node = 'test_qnse_adapter_retains_actual_equivalent_roughness_outputs'
source = inspect.getsource(getattr(module,name))
assert source.count(needle) == 1
tree = ast.parse(source.replace(needle,replacement))
exec(compile(tree, str(root/'proofs/t_c03rain/history_mutant.py')+':'+mode, 'exec'), module.__dict__)
raise SystemExit(pytest.main(['-q',str(root/'tests/test_v040_qnse_coupling.py')+'::'+node,
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/t-c03rain/pytest','--junitxml',str(output)]))
