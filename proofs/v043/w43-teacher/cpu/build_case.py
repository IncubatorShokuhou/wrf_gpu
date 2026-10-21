"""Build one original CPU-WRF 72 h case dir from a STAGED input set (lane w43-teacher) — CPU only, no JAX, no ALISIOS read.

usage: python3 build_case.py STAGED_RUN_DIR CASE_DIR NPROC_X NPROC_Y
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
(case / "input" / "namelist.input").write_text(new)
diff = [l for l in difflib.unified_diff(text.splitlines(), new.splitlines(), lineterm="", n=0)
        if l[:1] in "+-" and not l.startswith(("+++", "---"))]
assert all(re.match(r"^[+-]\s*(nproc_x|nproc_y|restart_interval)\s*=", l) for l in diff), diff
json.dump(dict(schema="w43-teacher.cpu_case.v1", staged_run_dir=str(src), ranks=nx * ny, nproc=[nx, ny], inputs=rows,
               namelist=dict(staged_sha256=sha(src / "namelist.input"), case_sha256=sha(case / "input" / "namelist.input"),
                             diff=diff)), open(case / "CASE.json", "w"), indent=1)
print(json.dumps({"case": str(case), "ranks": nx * ny, "diff": diff}))
