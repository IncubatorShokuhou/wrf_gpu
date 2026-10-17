"""List fusions of an after_optimizations HLO dump whose body touches a keyword (op_name), with kind and size.

usage: python scan_fusions.py HLO_TXT KEYWORD OUT_JSON   (E41 HLO unit: kInput transpose fusion > 500 instr = FAIL)
"""
import json, re, sys

hlo, key, out = sys.argv[1:4]
text = open(hlo).read()
comps = {}
for m in re.finditer(r'^%([\w.\-]+) \(.*?\{\n(.*?)\n\}\n', text, re.M | re.S):
    body = m.group(2).splitlines()
    comps[m.group(1)] = dict(n=sum(1 for l in body if re.search(r'= [^ ]+ [\w\-]+\(', l)),
                             hit=sum(1 for l in body if key in l), transposes=sum(1 for l in body if ' transpose(' in l),
                             f64=sum(1 for l in body if re.search(r'= f64\[', l)))
rows = []
for m in re.finditer(r'^\s*%([\w.\-]+) = (\(.*?\)|\S+) fusion\((.*?)\), kind=(\w+), calls=%([\w.\-]+)', text, re.M):
    c = comps.get(m.group(5))
    if c and c['hit']:
        outs = m.group(2).count('[') if m.group(2).startswith('(') else 1
        rows.append(dict(fusion=m.group(1), kind=m.group(4), computation=m.group(5), outputs=outs,
                         inputs=len([x for x in m.group(3).split(', ') if x.startswith('%')]), **c))
rows.sort(key=lambda r: -r['n'])
gate = [r for r in rows if r['kind'] == 'kInput' and r['transposes'] and r['n'] > 500]
json.dump(dict(hlo=hlo, keyword=key, fusions=rows, e41_hlo_fail=gate), open(out, 'w'), indent=1)
print('fusions touching', key, len(rows), 'total instr', sum(r['n'] for r in rows))
for r in rows[:15]:
    print(f"{r['n']:6d} {r['kind']:8s} out{r['outputs']:3d} in{r['inputs']:3d} T{r['transposes']:3d} {r['fusion']}")
print('E41 HLO kInput-transpose>500:', [(r['fusion'], r['n']) for r in gate])
