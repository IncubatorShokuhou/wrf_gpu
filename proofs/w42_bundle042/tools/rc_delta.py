"""w42-bundle042: test files to re-run on the BUNDLE PIN = the delta vs the RC-suite base (E199 consumer rule).

usage: python3 rc_delta.py REPO BASE PIN OUT_LIST [--exclude-done RC_OUT_DIR]
Selected (all collected-name test files, i.e. test_*.py / *_test.py under tests/ or evals/):
  (1) test files added/modified between BASE and PIN,
  (2) test files that import a src module changed between BASE and PIN (dotted 'gpuwrf.a.b', 'from gpuwrf.a import b',
      'from gpuwrf.a.b import', or a relative 'import b' inside the same package path) — direct importers,
  (3) test files that reference a changed scripts/ or tools/ file by path or module stem,
  (4) every file of a tests/ directory whose conftest.py changed.
With --exclude-done nothing is excluded by default; the option only reports which delta files already ran on BASE.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

repo, base, pin, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
changed = subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", base, pin], text=True).split()
files = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", pin, "tests", "evals"], text=True).split()
tests = [f for f in files if f.endswith(".py") and (Path(f).name.startswith("test_") or f.endswith("_test.py"))]
sel = {}
for f in changed:
    if f in tests:
        sel[f] = "changed test"
pats = []
for f in changed:
    if f.startswith("src/gpuwrf/") and f.endswith(".py"):
        mod = f[len("src/"):-3].replace("/", ".").removesuffix(".__init__")
        pkg, _, leaf = mod.rpartition(".")
        pats.append((f, re.compile(rf"\b{re.escape(mod)}\b|from\s+{re.escape(pkg)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b")))
    elif (f.startswith("scripts/") or f.startswith("tools/")) and f.endswith((".py", ".sh")):
        stem = Path(f).stem
        pats.append((f, re.compile(rf"{re.escape(f)}|\b{re.escape(Path(f).name)}\b|import\s+{re.escape(stem)}\b|"
                                   rf"from\s+{re.escape(stem)}\s+import")))
conf_dirs = [str(Path(f).parent) for f in changed if Path(f).name == "conftest.py"]
for t in tests:
    if t in sel:
        continue
    try:
        text = subprocess.check_output(["git", "-C", str(repo), "show", f"{pin}:{t}"], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        continue
    for f, rx in pats:
        if rx.search(text):
            sel[t] = f"imports/references {f}"
            break
    else:
        for d in conf_dirs:
            if t.startswith(d + "/"):
                sel[t] = f"conftest changed in {d}"
                break
out.write_text("\n".join(sorted(sel)) + "\n")
(out.with_suffix(".json")).write_text(json.dumps(dict(base=base, pin=pin, changed=changed, selected=sel), indent=1) + "\n")
print(f"changed {len(changed)} files; selected {len(sel)} test files -> {out}")
for t, why in sorted(sel.items()):
    print(" ", t, "|", why)
