"""Attribute a fusion body's instructions to source stack frames.
Usage: fusion_sources.py HLO_TXT FUSION_NAME [FUSION_NAME...]
Reads HLO_TXT + '.pb' (HloModuleProto) for stack_frame_index (field 17)."""
import sys, re, collections, json
from pathlib import Path
sys.path.insert(0, "src")
from gpuwrf.runtime.aot_executable import _proto_fields

hlo = Path(sys.argv[1]).read_text()
proto = Path(sys.argv[1] + ".pb").read_bytes()
sfi = [v for f, w, v in _proto_fields(proto) if f == 17]
assert len(sfi) == 1, len(sfi)
files, funcs, locs, frames = [], [], [], []
for f, w, v in _proto_fields(sfi[0]):
    if f == 1: files.append(v.decode())
    elif f == 2: funcs.append(v.decode())
    elif f == 3:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}
        locs.append((d.get(1, 0), d.get(2, 0), d.get(3, 0)))
    elif f == 4:
        d = {ff: vv for ff, ww, vv in _proto_fields(v)}
        frames.append((d.get(1, 0), d.get(2, 0)))

def chain(fid, depth=12):
    out = []
    while fid and len(out) < depth:
        loc_id, parent = frames[fid - 1]
        fn, fu, line = locs[loc_id - 1]
        f = files[fn - 1] if fn else "?"
        f = re.sub(r".*/src/gpuwrf/", "", f)
        out.append(f"{f}:{line} {funcs[fu - 1] if fu else '?'}")
        fid = parent
    return out

# split computations
comps = {}
cur = None
for line in hlo.splitlines():
    if cur is None:
        m = re.match(r"^\s*(?:ENTRY\s+)?%?([\w.\-]+)\s*(?:\(.*)?\{\s*$", line)
        if m and not line.startswith("HloModule"):
            cur, body = m.group(1), []
        continue
    if line.strip() == "}":
        comps[cur] = body; cur = None
    else:
        body.append(line)

calls = {}
for name, body in comps.items():
    for line in body:
        m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*\bfusion\(.*calls=%?([\w.\-]+)", line)
        if m: calls[m.group(1)] = (name, m.group(2), line)

report = {}
for fusion in sys.argv[2:]:
    parent, comp, call_line = calls[fusion]
    body = comps[comp]
    ops = collections.Counter()
    srcs = collections.Counter(); leaf = collections.Counter()
    for line in body:
        m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*\S+\s+([a-z][a-z0-9-]*)\(", line) or re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*\(.*?\)\s+([a-z][a-z0-9-]*)\(", line)
        op = m.group(2) if m else "?"
        ops[op] += 1
        s = re.search(r"stack_frame_id=(\d+)", line)
        if s:
            ch = chain(int(s.group(1)))
            leaf[ch[0]] += 1
            srcs[" <- ".join(ch[:4])] += 1
        else:
            leaf["<no frame>"] += 1
    print(f"=== {fusion}  comp={comp}  n_instr={len(body)} parent={parent}")
    print("ops:", dict(ops.most_common(25)))
    print("-- innermost frames:")
    for k, v in leaf.most_common(40): print(f"{v:5d}  {k}")
    print("-- 4-frame chains:")
    for k, v in srcs.most_common(30): print(f"{v:5d}  {k}")
    report[fusion] = {"comp": comp, "n_instr": len(body), "ops": dict(ops), "leaf": dict(leaf), "chains": dict(srcs)}
Path(sys.argv[1] + ".fusion_sources.json").write_text(json.dumps(report, indent=1))
