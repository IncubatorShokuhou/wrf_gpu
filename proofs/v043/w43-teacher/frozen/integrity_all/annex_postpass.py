"""Reference implementation (mass-opus) of the GATES_FROZEN EXPLAINED-trace ANNEX (G-INT (i)-(iv) + A1 + A2), post-pass over wn3's
frozen integrity classes (REGISTRY_ALLFRAMES 2a11c0cb). For every NEW item 'dXX FIELD@YYYY-MM-DD_HH': flagged cells = presence
mismatch (CPU != 0 xor GPU != 0; 3-D fields per column, any level). Explained iff ALL: (i) field in the trace family; (ii) GPU exactly 0
at every flagged cell; (iii) CPU max over flagged cells <= cap (ACSNOM: per-frame CPU increment <= 1e-3 kg/m2 AND value <= 2.5e-2;
TAUSS <= 5e-2); (iv) <= 200 flagged cells. Anything else = REAL FAIL. E200: operands finite.
usage: annex_postpass.py <classes_all_72h.json> <CPU dir> <GPU wrfout dir>"""
import glob, json, re, sys
from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
CAP = {'QICE': 1e-10, 'QSNOW': 1e-10, 'QGRAUP': 1e-10, 'QNICE': 10.0, 'QNSNOW': 10.0, 'SNOW': 1e-3, 'SNOWH': 1e-5, 'SNOWC': 1e-3,
       'TAUSS': 5e-2, 'QSNOWXY': 1e-6, 'CANLIQ': 1e-4, 'CANWAT': 1e-4, 'CANICE': 1e-4, 'FWET': 0.05, 'EVC': 0.01, 'ECAN': 1e-6,
       'QRAINXY': 1e-7, 'ACSNOM': 2.5e-2}
jf, C, G = sys.argv[1:4]
J = json.load(open(jf))
def rd(d, dom, v, t):
    f = [x for x in glob.glob(f'{d}/wrfout_{dom}_{t}:*') if not x.endswith('.tmp')]
    assert len(f) == 1, (d, dom, t, f)
    with Dataset(f[0]) as ds:
        a = np.asarray(ds[v][0], float)
    assert np.isfinite(a).all(), (d, dom, v, t); return a
items = J.get('new', [])
print(f'{jf.split("/")[-3]}: registry verdict {J.get("verdict")}; NEW items {len(items)}')
ok_all = True
for it in items:
    m = re.match(r'(d0\d) ([A-Z0-9_]+)(?:\(companion\))?@(\d{4}-\d\d-\d\d_\d\d)', it)
    if not m:
        print('  UNPARSED -> REAL FAIL:', it[:120]); ok_all = False; continue
    dom, fld, t = m.groups()
    if fld not in CAP:
        print(f'  {dom} {fld}@{t}: (i) outside trace family -> REAL FAIL'); ok_all = False; continue
    c, g = rd(C, dom, fld, t), rd(G, dom, fld, t)
    if c.ndim == 3:
        cc, gg = np.abs(c).max(0), np.abs(g).max(0)
    else:
        cc, gg = np.abs(c), np.abs(g)
    flag = (cc != 0) ^ (gg != 0)
    n = int(flag.sum()); gpu0 = bool((gg[flag] == 0).all()); cmax = float(cc[flag].max()) if n else 0.0
    ok3 = cmax <= CAP[fld]
    extra = ''
    if fld == 'ACSNOM':
        tp = (datetime.strptime(t, '%Y-%m-%d_%H') - timedelta(hours=1)).strftime('%Y-%m-%d_%H')
        cprev = np.abs(rd(C, dom, fld, tp)); inc = float((cc - cprev)[flag].max()) if n else 0.0
        ok3 = ok3 and inc <= 1e-3; extra = f' increment {inc:.2e}'
    ok = gpu0 and ok3 and n <= 200
    ok_all &= ok
    print(f'  {dom} {fld:8s}@{t}: flagged {n:4d} (iv {n <= 200}) | GPU exact 0 {gpu0} (ii) | CPU max {cmax:.2e} cap {CAP[fld]:.0e}{extra} (iii {ok3}) -> '
          + ('EXPLAINED' if ok else 'REAL FAIL'))
print('ANNEX', 'PASS (all NEW items explained)' if ok_all else 'FAIL')
