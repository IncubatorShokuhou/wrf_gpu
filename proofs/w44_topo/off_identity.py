"""Canonical release trace, with explicit case paths and lane-local temporary files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--case', required=True, choices=('swiss', 'prod', 'wn3'))
    p.add_argument('--topo-off', action='store_true')
    p.add_argument('--expected-sha')
    args = p.parse_args()
    cases = {'swiss': Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'),
             'prod': Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725'),
             'wn3': Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')}
    expected = {'swiss': 'b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a',
                'prod': '321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b',
                'wn3': None}
    assert (cases[args.case] / 'namelist.input').is_file()
    lane = Path('<USER_HOME>/wrf_gpu2_lanes/w44-topo/identity_tmp')
    lane.mkdir(exist_ok=True)
    script = args.tree / 'proofs/s1_port/release_identity.py'
    original = script.read_text()
    source = original.replace('<USER_HOME>/wrf_gpu2_lanes/s1-port', str(lane))
    assert source != original, 'canonical temp-parent hook changed'
    arm_name = 'release'
    if args.topo_off:
        arm_name = 'topo_shading=0'
        before = 'sys.argv = [str(probe), str(root), str(output), "release"]'
        after = 'sys.argv = [str(probe), str(root), str(output), "topo_shading=0"]'
        assert before in source, 'canonical arm hook changed'
        source = source.replace(before, after)
    sys.argv = [str(script), str(args.tree), str(args.out), str(cases[args.case])]
    exec(compile(source, str(script), 'exec'), {'__file__': str(script), '__name__': '__main__'})
    result = json.loads(args.out.read_text())
    arm = result['arms'][arm_name]
    reference = args.expected_sha if args.expected_sha is not None or args.topo_off else expected[args.case]
    valid = 'error' not in arm and isinstance(arm.get('sha'), str) and len(arm['sha']) == 64
    result['off_identity_gate'] = {'case': args.case, 'arm': arm_name, 'expected': reference,
        'actual': arm.get('sha'), 'trace_valid': valid,
        'passed': valid and arm.get('sha') == reference if reference is not None else None,
        'canonical_probe_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'only_host_override': 'temporary case parent moved into lane directory'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    assert valid, result['off_identity_gate']
    if reference is not None:
        assert result['off_identity_gate']['passed'], result['off_identity_gate']
    print(json.dumps(result['off_identity_gate']), flush=True)


if __name__ == '__main__':
    main()
