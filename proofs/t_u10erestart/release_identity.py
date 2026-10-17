"""Run the unchanged official identity probe with lane-local temporary cases."""
from pathlib import Path
import json
import os
import runpy
import sys
import tempfile

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
prod = len(sys.argv) > 3
official = root/'proofs/s1_port/release_identity.py'
original = tempfile.TemporaryDirectory


def temporary(*args, **kwargs):
    if str(kwargs.get('dir')) == '<USER_HOME>/wrf_gpu2_lanes/s1-port':
        kwargs['dir'] = output.parent
    return original(*args, **kwargs)


tempfile.TemporaryDirectory = temporary
sys.argv[0] = str(official)
runpy.run_path(str(official), run_name='__main__')
result = json.loads(output.read_text())
expected = ('6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b'
            if prod else
            'f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d')
assert result['arms']['release']['sha'] == expected
result['lane_temporary_case_root'] = str(output.parent)
output.write_text(json.dumps(result, indent=2)+'\n')
