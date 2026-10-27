"""Bind OFF/ON first-root AOT addresses to the actual rich E41 HLO receipts."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(run, gate, directory):
    assert gate['e41_gate'] == 'PASS', 'native/HLO gate must be complete and passing'
    result = {}
    for label in ('off', 'on'):
        programs = run[label + '_aot']['domains']
        assert programs
        rows = {}
        for name, status in programs.items():
            key = status.get('cheap_key')
            assert key, (label, name, 'missing actual AOT address')
            matches = []
            for entry in gate['entries']:
                paths = [entry['path'], *entry.get('aliases', [])]
                if entry.get('program') == name and any(
                        key in Path(p).parts or Path(p).stem in (key, 'k_' + key) for p in paths):
                    matches.append(entry)
            assert len(matches) == 1, (label, name, key, 'ambiguous/missing E41 artifact')
            entry = matches[0]
            assert entry['kinput_transpose_status'].startswith('MEASURED_RICH_')
            token = hashlib.sha256(entry['path'].encode()).hexdigest()[:12]
            hlo = json.loads((directory / ('hlo_scan_' + token + '.json')).read_text())
            transpose = hlo['transpose_hero_fusions']
            rows[name] = {'cheap_key': key, 'blob_sha256': entry['blob_sha256'],
                          'n_fusions': hlo['n_fusions'], 'transpose_fusions': len(transpose),
                          'transpose_instructions': sum(row['n_transpose'] for row in transpose),
                          'max_kinput_instructions': max((row['n_instr'] for row in transpose
                                                         if row['kind'] == 'kInput'), default=0)}
        result[label] = rows
    assert result['off'].keys() == result['on'].keys(), 'different resolved programs'
    result['on_minus_off'] = {name: {field: result['on'][name][field] - result['off'][name][field]
                                     for field in ('n_fusions', 'transpose_fusions', 'transpose_instructions')}
                              for name in result['off']}
    result['scope'] = 'Actual first-root compiled program structure; runtime device work is in profile/metrics.json'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('e41', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.run.read_text()), json.loads((args.e41 / 'wave_gate.json').read_text()), args.e41)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
