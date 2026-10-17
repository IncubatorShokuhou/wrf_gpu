import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, '<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/C11R/snapshot/scripts/v025')
import cpu_guard
import pytest
from gpuwrf.coupling import scan_adapters as sa

root = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/C11R')
source = inspect.getsource(sa.wdm6_adapter)
original = sa.wdm6_adapter.__code__
old = '_small_grid_firewall(mp(state.Nn), (ny, nx)), mp(state.Nc), mp(state.Nr),'
assert source.count(old) == 1
rows = []
for label, new, test in [
    ('removed', 'mp(state.Nn), mp(state.Nc), mp(state.Nr),', 'test_small_grid_fences_exactly_the_nn_column_view'),
    ('wrong_operand', 'mp(state.Nn), _small_grid_firewall(mp(state.Nc), (ny, nx)), mp(state.Nr),', 'test_small_grid_fences_exactly_the_nn_column_view'),
    ('unconditional', 'jax.lax.optimization_barrier(mp(state.Nn)), mp(state.Nc), mp(state.Nr),', 'test_large_grid_traces_no_barrier'),
]:
    namespace = dict(sa.__dict__)
    exec(compile(source.replace(old, new), '<rv-c11-mutant-' + label + '>', 'exec'), namespace)
    sa.wdm6_adapter.__code__ = namespace['wdm6_adapter'].__code__
    xml = root / ('mutant_' + label + '.xml')
    try:
        rc = pytest.main(['-q', '-p', 'no:cacheprovider', '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/pytest', '--junitxml=' + str(xml), 'tests/test_t_cfix_cu_wdm6_small_grid_firewall.py::' + test])
    finally:
        sa.wdm6_adapter.__code__ = original
    tree = ET.parse(xml)
    failures = tree.findall('.//failure')
    errors = tree.findall('.//error')
    killed = rc == 1 and len(failures) == 1 and not errors and 'AssertionError' in ''.join(failures[0].itertext())
    rows.append(dict(label=label, rc=int(rc), failures=len(failures), errors=len(errors), genuine_assertion=bool(killed)))
    assert killed, rows[-1]
(root / 'mutants.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows))
