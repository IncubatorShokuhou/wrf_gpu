"""Build the #148 CPU twin namelist from the reference 9-domain namelist (a2, nproc 2x2).

mode 'n18'  : pure decomposition twin — ONLY nproc_x/nproc_y (+ restart_interval) change.
mode 'n18k5': domains KEEP = d01,d02,d04,d05,d06 renumbered 1..5 (all 15 failing #148 D6 rows live there; feedback=0), plus the
              decomposition change. Every per-domain key with exactly max_dom values is subset to KEEP; single-value keys stay;
              any other length refuses (no guessing). grid_id -> 1..N, parent_id -> parents remapped, max_dom -> N.

usage: python make_twin_namelist.py <reference namelist> <out namelist> <mode> <nproc_x> <nproc_y>
prints a JSON receipt (changed keys, domain map); the caller writes the unified diff next to the run.
"""
import json
import re
import sys

KEEP = [1, 2, 4, 5, 6]


def parse_values(text):
    return [v.strip() for v in text.strip().rstrip(",").split(",")]


def build(ref_text, mode, nproc_x, nproc_y):
    m = re.search(r"(?m)^\s*max_dom\s*=\s*(\d+)\s*,", ref_text)
    max_dom = int(m.group(1))
    pid = parse_values(re.search(r"(?m)^\s*parent_id\s*=\s*(.*)$", ref_text).group(1))
    assert max_dom == 9 and len(pid) == 9, "reference must be the 9-domain #148 namelist"
    keep = KEEP if mode == "n18k5" else list(range(1, 10))
    new_index = {old: i + 1 for i, old in enumerate(keep)}
    changed, out_lines = {}, []
    for line in ref_text.splitlines(keepends=True):
        mm = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*?)(\s*)$", line.rstrip("\n"))
        if not mm or line.lstrip().startswith(("&", "/", "!")):
            out_lines.append(line)
            continue
        indent, key, eq, rhs, _ = mm.groups()
        vals = parse_values(rhs)
        new = None
        if key == "nproc_x":
            new = [str(nproc_x)]
        elif key == "nproc_y":
            new = [str(nproc_y)]
        elif key == "restart_interval":
            new = ["100000"]
        elif mode == "n18k5":
            if key == "max_dom":
                new = [str(len(keep))]
            elif key == "grid_id":
                new = [str(i + 1) for i in range(len(keep))]
            elif key == "parent_id":
                new = [str(new_index[int(pid[k - 1])]) if k > 1 else "1" for k in keep]
            elif len(vals) == max_dom:
                new = [vals[k - 1] for k in keep]
            elif len(vals) != 1:
                raise ValueError(f"per-domain key {key!r} has {len(vals)} values (not 1 or {max_dom}): refuse")
        if new is not None and new != vals:
            changed[key] = {"old": vals, "new": new}
            line = f"{indent}{key}{eq}{', '.join(new)},\n"
        out_lines.append(line)
    return "".join(out_lines), changed, {f"d{k:02d}": f"d{new_index[k]:02d}" for k in keep}


if __name__ == "__main__":
    ref, out, mode, nx, ny = sys.argv[1:6]
    text, changed, dmap = build(open(ref).read(), mode, int(nx), int(ny))
    open(out, "x").write(text)
    print(json.dumps({"mode": mode, "domain_map_ref_to_twin": dmap, "changed": changed}, indent=1))
