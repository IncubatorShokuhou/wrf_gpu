"""Multiset diff of canonicalized fused computations between two after_optimizations HLO dumps.

Canonical form: instruction names -> positional ids, metadata/backend_config dropped, computation name dropped.
usage: python canon_diff.py BASE_HLO NEW_HLO OUT_JSON
"""
import collections, hashlib, json, re, sys

def canon(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r'^%(fused_[\w.\-]*|[\w.\-]*fusion[\w.\-]*) \((.*?)\{\n(.*?)\n\}\n', text, re.M | re.S):
        ids = {}
        lines = []
        for l in m.group(3).splitlines():
            l = re.sub(r', metadata=\{.*?\}(?=,|$)', '', l)
            l = re.sub(r', backend_config=.*$', '', l)
            def rep(mm):
                ids.setdefault(mm.group(1), f'v{len(ids)}')
                return '%' + ids[mm.group(1)]
            lines.append(re.sub(r'%([\w.\-]+)', rep, l).strip())
        body = '\n'.join(lines)
        out[m.group(1)] = (hashlib.sha256(body.encode()).hexdigest()[:16], len(lines), body)
    return out

a, b = canon(sys.argv[1]), canon(sys.argv[2])
ca = collections.Counter(h for h, _, _ in a.values()); cb = collections.Counter(h for h, _, _ in b.values())
only_a = ca - cb; only_b = cb - ca
name = lambda d, h: sorted(((k, n) for k, (hh, n, _) in d.items() if hh == h), key=lambda x: -x[1])
res = dict(base=sys.argv[1], new=sys.argv[2], base_computations=len(a), new_computations=len(b),
           identical=sum((ca & cb).values()),
           only_base=[dict(hash=h, count=c, names=name(a, h)[:3]) for h, c in only_a.items()],
           only_new=[dict(hash=h, count=c, names=name(b, h)[:3]) for h, c in only_b.items()])
json.dump(res, open(sys.argv[3], 'w'), indent=1)
print('computations', len(a), len(b), 'identical', res['identical'], 'only_base', sum(only_a.values()), 'only_new', sum(only_b.values()))
for side, rows in (('BASE', res['only_base']), ('NEW', res['only_new'])):
    for r in sorted(rows, key=lambda r: -r['names'][0][1])[:12]: print(' ', side, r['names'][0], r['count'])
