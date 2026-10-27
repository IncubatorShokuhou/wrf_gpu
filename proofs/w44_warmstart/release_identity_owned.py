"""Run the unchanged official identity probe with lane-owned temporary files."""
import json
from pathlib import Path
import runpy
import sys
import tempfile


def main():
    source, output = map(Path, sys.argv[1:3])
    mode = sys.argv[3]
    assert mode in ('swiss', 'prod')
    original = tempfile.TemporaryDirectory
    own = Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart/identity_tmp')
    own.mkdir(exist_ok=True)
    def temporary(*args, **kwargs):
        if kwargs.get('dir') == Path('<USER_HOME>/wrf_gpu2_lanes/s1-port'):
            kwargs['dir'] = own
        return original(*args, **kwargs)
    tempfile.TemporaryDirectory = temporary
    script = source / 'proofs/s1_port/release_identity.py'
    sys.argv = [str(script), str(source), str(output)]
    if mode == 'prod':
        sys.argv.append('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        tempfile.TemporaryDirectory = original
    result = json.loads(output.read_text())
    record = result['arms']['release']
    expected = 'b977712f' if mode == 'swiss' else '321aa651'
    assert record.get('sha', '').startswith(expected), record
    result['owned_wrapper'] = {'only_change': 'official temporary directory redirected to own lane',
                               'expected_prefix': expected, 'passed': True}
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
