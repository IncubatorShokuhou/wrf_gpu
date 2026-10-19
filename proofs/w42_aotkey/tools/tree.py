"""tree.py ROOT OUT.json: every path under ROOT with SHA-256 (dirs as 'dir')."""
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
tree = {p.relative_to(root).as_posix(): "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))}
Path(sys.argv[2]).write_text(json.dumps(tree, indent=0, sort_keys=True))
print(len(tree), "paths", hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16])
