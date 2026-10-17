"""Established release identity probe, with its temporary directory lane-local."""
import json
from pathlib import Path
import runpy
import sys
import tempfile

root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
original = tempfile.TemporaryDirectory


def local_directory(*args, **kwargs):
    if str(kwargs.get('dir', '')) == '<USER_HOME>/wrf_gpu2_lanes/s1-port':
        kwargs['dir'] = output.parent
    return original(*args, **kwargs)


tempfile.TemporaryDirectory = local_directory
runpy.run_path(str(root / 'proofs/s1_port/release_identity.py'), run_name='__main__')
result = json.loads(output.read_text())
# Upstream replaces sys.argv; use the case recorded in its receipt.
expected = ('6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b'
            if 's0_case_20260725' in result['case'] else
            'f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d')
assert result['arms']['release']['sha'] == expected, result['arms']['release']
