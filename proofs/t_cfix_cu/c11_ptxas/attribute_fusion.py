"""Attribute one fusion of an XLA after_optimizations HLO text dump to source lines (stack frames).

usage: python attribute_fusion.py HLO_TXT FUSION_INSTR_NAME OUT_JSON
"""
import collections, json, re, sys

hlo, fusion, out = sys.argv[1:4]
text = open(hlo).read()
lines = text.splitlines()

def table(name, nxt):
    i = lines.index(name) + 1; rows = {}
    while i < len(lines) and lines[i] and lines[i] not in nxt:
        k, v = lines[i].split(' ', 1); rows[int(k)] = v; i += 1
    return rows
heads = {'FileNames', 'FunctionNames', 'FileLocations', 'StackFrames'}
files = {k: v.strip('"') for k, v in table('FileNames', heads).items()}
funcs = {k: v.strip('"') for k, v in table('FunctionNames', heads).items()}
locs = {}
for k, v in table('FileLocations', heads).items():
    d = dict(re.findall(r'(\w+)=(\d+)', v)); locs[k] = (files[int(d['file_name_id'])], funcs[int(d['function_name_id'])], int(d['line']))
frames = {}
for k, v in table('StackFrames', heads | {''}).items():
    d = dict(re.findall(r'(\w+)=(\d+)', v)); frames[k] = (int(d['file_location_id']), int(d.get('parent_frame_id', 0)))

def stack(fid):
    out = []
    while fid and fid in frames:
        loc, parent = frames[fid]; out.append(locs[loc])
        if parent == fid: break
        fid = parent
    return out  # innermost first

m = re.search(r'^\s*%' + re.escape(fusion) + r' = .*? calls=%([\w.\-]+)', text, re.M)
callee = m.group(1)
start = text.index('\n%' + callee + ' ')
end = text.index('\n}\n', start)
body = text[start:end].splitlines()[1:]
ops = collections.Counter(); inner = collections.Counter(); chains = collections.Counter()
n = 0
for l in body:
    mm = re.match(r'\s*(?:ROOT )?%[\w.\-]+ = (\S+?)\[?[^ ]* (\w[\w\-]*)\(', l)
    op = re.search(r'= [^ ]+ ([\w\-]+)\(', l)
    if not op: continue
    n += 1; ops[op.group(1)] += 1
    sf = re.search(r'stack_frame_id=(\d+)', l)
    if sf:
        st = [f for f in stack(int(sf.group(1))) if '/gpuwrf/' in f[0]]
        if st:
            inner[f'{st[0][0]}:{st[0][2]} ({st[0][1]})'] += 1
            chains[' <- '.join(f'{f[0].split("/gpuwrf/")[-1]}:{f[2]}' for f in st[:6])] += 1
# callees (nested fusions/computations called from the body)
calls = collections.Counter(re.findall(r'(?:calls|to_apply)=%([\w.\-]+)', '\n'.join(body)))
res = dict(hlo=hlo, fusion=fusion, computation=callee, instructions=n, ops=ops.most_common(),
           innermost_source=inner.most_common(), stack_chains=chains.most_common(60), callees=calls.most_common(20),
           params=re.findall(r'(param_\d+\.\d+): ([^,)]+\])', text[start:text.index('\n', start+1)]))
json.dump(res, open(out, 'w'), indent=1)
print(callee, n, ops.most_common(12))
for k, v in inner.most_common(40): print(v, k)
