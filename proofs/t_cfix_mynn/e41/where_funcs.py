"""Which computations/fusions contain instructions whose stack chain includes given function names.
Usage: where_funcs.py HLO_TXT FUNC..."""
import sys, re, collections
from pathlib import Path
sys.path.insert(0, "src")
from gpuwrf.runtime.aot_executable import _proto_fields
hlo = Path(sys.argv[1]).read_text().splitlines()
proto = Path(sys.argv[1] + ".pb").read_bytes()
sfi = [v for f, w, v in _proto_fields(proto) if f == 17][0]
files, funcs, locs, frames = [], [], [], []
for f, w, v in _proto_fields(sfi):
    if f == 1: files.append(v.decode())
    elif f == 2: funcs.append(v.decode())
    elif f == 3:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}; locs.append((d.get(1, 0), d.get(2, 0), d.get(3, 0)))
    elif f == 4:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}; frames.append((d.get(1, 0), d.get(2, 0)))
memo = {}
def fnames(fid):
    if fid in memo: return memo[fid]
    out = set(); f0 = fid
    while fid:
        loc_id, parent = frames[fid - 1]; fn, fu, line = locs[loc_id - 1]
        out.add(funcs[fu - 1]); fid = parent
    memo[f0] = out; return out
targets = sys.argv[2:]
comp = None; hits = collections.defaultdict(collections.Counter); sizes = collections.Counter()
for line in hlo:
    if comp is None:
        m = re.match(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\(.*)?\{\s*$", line)
        if m and not line.startswith("HloModule"): comp = m.group(1)
        continue
    if line.strip() == "}": comp = None; continue
    sizes[comp] += 1
    s = re.search(r"stack_frame_id=(\d+)", line)
    if not s: continue
    names = fnames(int(s.group(1)))
    for t in targets:
        if t in names: hits[t][comp] += 1
callers = {}
for line in hlo:
    m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*\bfusion\(.*calls=%?([\w.\-]+)", line)
    if m: callers[m.group(2)] = m.group(1)
for t in targets:
    print("==", t)
    for c, n in hits[t].most_common(25):
        print(f"   {n:4d} in {c} (size {sizes[c]}) fusion={callers.get(c, '-')}")
