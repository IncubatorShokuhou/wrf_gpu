"""Delete the one-moment saturation/freezing update; require a physics assertion."""
from pathlib import Path
import json, os, sys
from xml.etree import ElementTree

root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert (root / '.git').is_file(), 'Use an immutable detached worktree'
sys.path[:0] = [str(root / 'src'), str(root / 'scripts/v025')]
import cpu_guard
from gpuwrf.physics.nssl2mom import one_moment_adjustment
import pytest

def omitted(qx, cx, thetap, qwvp, temg, *args):
    return qx, cx, thetap, qwvp, temg

one_moment_adjustment.adjust = omitted
rc = pytest.main([
    str(root / 'tests/v040/test_nssl_one_moment.py') +
    '::test_complete_original_one_moment_column[4-fp64-21]',
    '-q', '--tb=short', '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/s2-mpmore/pytest',
    '--junitxml=' + str(output.with_suffix('.xml'))])
suite = ElementTree.parse(output.with_suffix('.xml')).getroot().find('testsuite')
assert rc == 1 and suite.get('failures') == '1' and suite.get('errors') == '0', (rc, suite.attrib)
failure = suite.find('testcase/failure')
assert 'AssertionError' in failure.get('message', ''), failure.get('message')
output.write_text(json.dumps(dict(killed=True, scope='one-moment saturation/freezing update',
                                 assertion=failure.get('message')), indent=2) + '\n')
