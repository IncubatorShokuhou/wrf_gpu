"""Compare optimized fusion bodies by alpha-normalized dataflow, never names.

Metadata and numeric HLO identifiers are excluded; operand order, parameter
numbers, shapes, layouts, constants, and operation attributes are preserved.
This is a structural compiler comparison, not numerical/GPU timing evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from e41.hlo_scan import HEADER, INSTR, OP, scan_text


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(block, external):
    nodes = []
    identifiers = {}
    for line in block:
        found = INSTR.match(line)
        if found:
            identifiers[found[1]] = f'i{len(nodes)}'
            nodes.append((found[1], found[2], line.lstrip().startswith('ROOT ')))
    canonical = []
    operators = Counter()
    for name, rhs, root in nodes:
        rhs = re.sub(r', metadata=\{[^}]*\}', '', rhs)
        rhs = re.sub(r'%([\w.\-]+)', lambda m: '%' + (identifiers[m[1]]
                     if m[1] in identifiers else 'computation_' + external(m[1])), rhs)
        operators[OP.search(rhs)[1]] += 1
        canonical.append(('ROOT ' if root else '') + identifiers[name] + ' = ' + rhs)
    return '\n'.join(canonical), dict(sorted(operators.items()))


def inventory(path):
    text = path.read_text()
    lines = text.splitlines()
    scan = scan_text(text)
    blocks = {}
    name = None
    for i, line in enumerate(lines):
        if name is None:
            head = HEADER.match(line)
            if head and not line.startswith('HloModule'):
                name, start = head[1], i
        elif line.strip() == '}':
            blocks[name] = lines[start:i + 1]
            name = None
    normalized = {}

    def external(name):
        if name not in normalized:
            normalized[name] = digest(normalize(blocks[name], external)[0])
        return normalized[name]

    rows = []
    for row in scan['transpose_hero_fusions']:
        begin, end = row['comp_lines']
        block = lines[begin - 1:end]
        canonical, operators = normalize(block, external)
        rows.append(dict(row, dag_sha256=digest(canonical), operators=operators,
                         constant_values_elided=('...' in canonical)))
    return dict(path=str(path), sha256=digest(text), n_fusions=scan['n_fusions'], rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ref, cand = inventory(args.reference), inventory(args.candidate)
    ref_dags = {}
    for row in ref['rows']:
        ref_dags.setdefault(row['dag_sha256'], []).append(row)
    oversized = []
    for row in cand['rows']:
        if row['kind'] == 'kInput' and row['n_instr'] > 500:
            oversized.append(dict(candidate=row, identical_reference=ref_dags.get(row['dag_sha256'], [])))
    report = dict(scope=__doc__, reference=ref, candidate=cand, oversized_matches=oversized)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(reference_max=max(r['n_instr'] for r in ref['rows'] if r['kind'] == 'kInput'),
                          candidate_max=max(r['n_instr'] for r in cand['rows'] if r['kind'] == 'kInput'),
                          oversized_matches=oversized), indent=2))


if __name__ == '__main__':
    main()
