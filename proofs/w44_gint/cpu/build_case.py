"""Build one original CPU-WRF 72 h case dir from a STAGED input set (lane w44-gint; copy of w43-teacher build_case.py + optional RUN_HOURS) — CPU only, no JAX, no ALISIOS read.

usage: python3 build_case.py STAGED_RUN_DIR CASE_DIR NPROC_X NPROC_Y [RUN_HOURS [MAX_DOM]]
(MAX_DOM 1 = root-only run: with feedback = 0 the root never sees its nests, so its trajectory is the 3-nest root trajectory;
 checked bitwise against the 3-nest reference by the caller)
STAGED_RUN_DIR holds namelist.input + wrfinput_d01..03 + wrfbdy_d01 + wrflowinp_d01..03 (copies made by stage.py; refused if
under an ALISIOS root). CASE_DIR/input gets real copies; the namelist differs from the staged one ONLY in nproc_x/nproc_y
(decomposition) and restart_interval -> 100000 (no restart files; numerics unchanged). CASE.json records SHA-256 of every
staged and copied file plus the exact line diff.
"""
import difflib, hashlib, json, re, shutil, sys
from pathlib import Path

ROOTS = ("<DATA_ROOT>/alisios/", "<DATA_ROOT>/server/work/src/alisios/", "<DATA_ROOT>/server/work/alisios/")
FILES = ("wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


src, case, nx, ny = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
hours = int(sys.argv[5]) if len(sys.argv) > 5 else 72
maxdom = int(sys.argv[6]) if len(sys.argv) > 6 else 3
assert maxdom in (1, 3)
assert not str(src).startswith(ROOTS), src
(case / "input").mkdir(parents=True, exist_ok=False)
rows = {}
for n in FILES:
    s = src / n
    assert s.is_file() and not str(s.resolve()).startswith(ROOTS), s
    shutil.copyfile(s, case / "input" / n)
    rows[n] = dict(staged=str(s), sha256=sha(s), copy_sha256=sha(case / "input" / n))
    assert rows[n]["sha256"] == rows[n]["copy_sha256"]
text = (src / "namelist.input").read_text()
new = text
for key, val in (("nproc_x", nx), ("nproc_y", ny), ("restart_interval", 100000)):
    new, k = re.subn(rf"^(\s*{key}\s*=\s*)[^,\n]*(,.*)$", rf"\g<1>{val}\g<2>", new, flags=re.M)
    assert k == 1, (key, k)
assert re.search(r"^\s*max_dom\s*=\s*3,", new, re.M) and re.search(r"^\s*run_hours\s*=\s*72,", new, re.M)
if hours != 72:   # WRF: nonzero run_hours takes precedence over end_* (numerics of the first RUN_HOURS unchanged)
    new, k = re.subn(r"^(\s*run_hours\s*=\s*)72(,.*)$", rf"\g<1>{hours}\g<2>", new, flags=re.M); assert k == 1
if maxdom != 3:
    assert re.search(r"^\s*feedback\s*=\s*0,", new, re.M), "root-only twin needs one-way nesting"
    new, k = re.subn(r"^(\s*max_dom\s*=\s*)3(,.*)$", rf"\g<1>{maxdom}\g<2>", new, flags=re.M); assert k == 1
(case / "input" / "namelist.input").write_text(new)
diff = [l for l in difflib.unified_diff(text.splitlines(), new.splitlines(), lineterm="", n=0)
        if l[:1] in "+-" and not l.startswith(("+++", "---"))]
assert all(re.match(r"^[+-]\s*(nproc_x|nproc_y|restart_interval|run_hours|max_dom)\s*=", l) for l in diff), diff
json.dump(dict(schema="w44-gint.cpu_case.v1", run_hours=hours, max_dom=maxdom, staged_run_dir=str(src), ranks=nx * ny, nproc=[nx, ny], inputs=rows,
               namelist=dict(staged_sha256=sha(src / "namelist.input"), case_sha256=sha(case / "input" / "namelist.input"),
                             diff=diff)), open(case / "CASE.json", "w"), indent=1)
print(json.dumps({"case": str(case), "ranks": nx * ny, "diff": diff}))
