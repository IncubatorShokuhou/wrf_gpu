"""Map retained actual-GPU E41 flags to rich-HLO source stacks on CPU."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def source_stacks(text):
    files, funcs, locs, frames = {}, {}, {}, {}
    section = None
    for line in text.splitlines():
        if line in ('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'):
            section = line
            continue
        if section and re.match(r'^\d+ ', line):
            if section in ('FileNames', 'FunctionNames'):
                i, value = re.match(r'^(\d+) "(.*)"', line).groups()
                (files if section == 'FileNames' else funcs)[i] = value
            elif section == 'FileLocations':
                m = re.match(r'^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+)', line)
                locs[m[1]] = (files[m[2]], funcs[m[3]], int(m[4]))
            else:
                m = re.match(r'^(\d+) \{file_location_id=(\d+)(?: parent_frame_id=(\d+))?', line)
                frames[m[1]] = (m[2], m[3])
        elif line.strip():
            section = None
    def chain(frame):
        result, seen = [], set()
        while frame in frames and frame not in seen:
            seen.add(frame)
            loc, frame = frames[frame]
            result.append(locs[loc])
        return result
    return chain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    import wave_cubin_gate as gate
    from e41 import hlo_scan
    import zstandard
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    verdict = json.loads((a.case / 'e41/wave_gate.json').read_text())
    record = next(r for r in verdict['entries'] if r['source_kind'] == 'jax_cache')
    pins = list((a.case / 'e41/pins').glob('*' + record['entry']))
    assert len(pins) == 1, pins
    raw = pins[0].read_bytes()
    blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 33)[4:]
    assert hashlib.sha256(blob).hexdigest() == record['blob_sha256']
    text = gate._optimized_hlo(blob, ax, asm)
    chain = source_stacks(text)
    scan = hlo_scan.scan_text(text)
    lines = text.splitlines()
    calls = {}
    for line in lines:
        match = re.match(r'\s*(?:ROOT )?%?([\w.\-]+) = .* fusion\(', line)
        if match:
            calls[match[1].replace('.', '_')] = (match[1], line)
    flags = {name: {'stack_B': size} for elf in record['elfs'] for name, size in elf['stack_over256']}
    for name, count in record['kinput_transpose_over500']:
        flags.setdefault(name.replace('.', '_'), {})['kinput_HLO'] = count
    for name, row in flags.items():
        hlo_name, call = calls[name]
        comp = re.search(r'calls=%?([\w.\-]+)', call)[1]
        start = next(i for i, line in enumerate(lines) if re.match(r'%?' + re.escape(comp) + r' ', line))
        end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == '}')
        sources, stacks = Counter(), Counter()
        for line in lines[start+1:end]:
            frame = re.search(r'stack_frame_id=(\d+)', line)
            stack = chain(frame[1]) if frame else []
            project = [f for f in stack if '/gpuwrf/' in f[0]]
            if project:
                sources[project[0]] += 1
                stacks[tuple(project)] += 1
        assert sources, (name, 'rich HLO has no recoverable project source stacks')
        row.update(hlo_fusion=hlo_name, comp=comp, instructions=end-start-1, call=call.strip(),
                   sources=[{'file': f, 'function': fn, 'line': ln, 'instructions': n}
                            for (f, fn, ln), n in sources.most_common(30)],
                   stacks=[{'frames': s, 'instructions': n} for s, n in stacks.most_common(15)])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'actual.optimized.hlo.gz').write_bytes(gzip.compress(text.encode(), mtime=0))
    assert flags, 'no flags: this attribution probe requires a failing E41 artifact'
    report = {'case': str(a.case), 'pin': str(pins[0]), 'pin_sha256': hashlib.sha256(raw).hexdigest(),
              'blob_sha256': record['blob_sha256'], 'hlo_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'flags': flags, 'scope': 'actual GPU artifact source attribution; runtime impact unknown'}
    (a.out / 'source_map.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({n: {**{k:v for k,v in r.items() if k not in ('stacks','call','sources')},
                          'sources': r['sources'][:8]} for n,r in flags.items()}, indent=2))


if __name__ == '__main__':
    main()
