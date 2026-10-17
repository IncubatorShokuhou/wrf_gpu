import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

root = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/C11R')
producer = Path('<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/c11')
scripts = root / 'snapshot/proofs/t_cfix_cu/c11_ptxas'
paths = [root / 'base.hlo', root / 'fix.hlo']
try:
    for name, path in zip(['hlo_5c15.txt.gz', 'hlo_fix.txt.gz'], paths):
        with gzip.open(producer / name, 'rb') as f:
            path.write_bytes(f.read())
    sys.argv = ['canon_diff.py', *map(str, paths), str(root / 'canonical.json')]
    result = runpy.run_path(str(scripts / 'canon_diff.py'))
    assert (len(result['a']), len(result['b']), result['res']['identical']) == (3314, 3315, 3309)
    changed = {}
    for side, path in zip(['base', 'new'], paths):
        text = path.read_text()
        lines = text.splitlines()
        def table(name):
            i = lines.index(name) + 1
            rows = {}
            while i < len(lines) and lines[i] and lines[i] not in {'FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'}:
                k, value = lines[i].split(' ', 1)
                rows[int(k)] = value
                i += 1
            return rows
        files = {k: v.strip('"') for k, v in table('FileNames').items()}
        funcs = {k: v.strip('"') for k, v in table('FunctionNames').items()}
        locs = {}
        for k, value in table('FileLocations').items():
            d = dict(re.findall(r'(\w+)=(\d+)', value))
            locs[k] = (files[int(d['file_name_id'])], funcs[int(d['function_name_id'])])
        frames = {}
        for k, value in table('StackFrames').items():
            d = dict(re.findall(r'(\w+)=(\d+)', value))
            frames[k] = (int(d['file_location_id']), int(d.get('parent_frame_id', 0)))
        def touches_wdm6(body):
            for fid in map(int, re.findall(r'stack_frame_id=(\d+)', body)):
                seen = set()
                while fid in frames and fid not in seen:
                    seen.add(fid)
                    loc, fid = frames[fid]
                    if any('wdm6' in value for value in locs[loc]):
                        return True
            return False
        comps = {m.group(1): m.group(3) for m in re.finditer(r'^%(fused_[\w.\-]*|[\w.\-]*fusion[\w.\-]*) \((.*?)\{\n(.*?)\n\}\n', text, re.M | re.S)}
        names = [pair[0] for row in result['res']['only_' + side] for pair in row['names']]
        changed[side] = {}
        for name in names:
            if touches_wdm6(comps[name]):
                changed[side][name] = 'WDM6 stack frames'
            else:
                body = comps[name]
                assert side == 'new' and name == 'fused_transpose.11'
                assert 'constant(1e+08)' in body and 'f32[44,42,42]{0,1,2} parameter(0)' in body
                assert 'ROOT ' in body and 'f64[44,1764]' in body and ' transpose(' in body
                changed[side][name] = 'Nn first-step 1e8 select + column view transpose + f64 convert; initialization/reshape frames, not WDM6 module frames'
        sys.argv = ['scan_fusions.py', str(path), '', str(root / ('scan_' + side + '.json'))]
        runpy.run_path(str(scripts / 'scan_fusions.py'))
    scans = [json.loads((root / ('scan_' + side + '.json')).read_text()) for side in ['base', 'new']]
    assert [len(s['e41_hlo_fail']) for s in scans] == [1, 0]
    native = json.loads((root / 'cubin_scan_independent.json').read_text())
    stack = [r for r in native if r['stack'] > 256]
    unchanged_stack = []
    for row in stack:
        comps = [next(r['computation'] for r in s['fusions'] if r['fusion'] == row['kernel']) for s in scans]
        assert result['a'][comps[0]][0] == result['b'][comps[1]][0]
        unchanged_stack.append(dict(kernel=row['kernel'], stack=row['stack'], canonical_hash=result['a'][comps[0]][0]))
    receipt = dict(ptx_sha256=hashlib.sha256((producer / 'after_nncol_module.ptx').read_bytes()).hexdigest(),
                   cubin_sha256=hashlib.sha256((root / 'after.cubin').read_bytes()).hexdigest(),
                   canonical=dict(base=3314, new=3315, same=3309, changed_wdm6=changed),
                   hlo_failures=[s['e41_hlo_fail'] for s in scans],
                   native_kernels=len(native), native_same_producer=native == json.loads((producer / 'cubin_scan_after.json').read_text()),
                   stack_failures=unchanged_stack, nonleaf_failures=sum(bool(r['nonleaf_over1k']) for r in native),
                   callgraph_unknown=sum(bool(r['callgraph_unknown']) for r in native),
                   scope='CPU retained HLO/PTX offline assembly. No original-WRF fidelity or GPU speed result.')
    (root / 'analysis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
finally:
    for path in paths:
        path.unlink(missing_ok=True)
