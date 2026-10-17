"""Attest CUDA-equivalent source and unshifted lines against the lane's base.

The normalization removes only import routing and Python capability branches
whose CUDA arm is the predecessor expression. It is not executable validation.
usage: python source_identity.py BASE_COMMIT OUTPUT_JSON
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
base, output = sys.argv[1], Path(sys.argv[2])
rows = []
paths = [*root.joinpath("src/gpuwrf/kernels").glob("*.py"),
         root / "src/gpuwrf/nesting/sint_kernel.py",
         root / "src/gpuwrf/coupling/boundary_apply.py"]
for path in sorted(paths):
    candidate = path.read_text()
    if "pallas_call(" not in candidate:
        continue
    rel = path.relative_to(root).as_posix()
    original = subprocess.check_output(["git", "show", f"{base}:{rel}"], cwd=root, text=True)
    normalized = candidate.replace("from gpuwrf.backends import pallas as pl",
                                   "from jax.experimental import pallas as pl")
    normalized = normalized.replace("if interpret or not pl.cuda_native():", "if interpret:")
    normalized = normalized.replace("from gpuwrf.backends.platform import cuda_native as pl_cuda_native", "")
    normalized = normalized.replace("jnp.minimum(lane // ng, ncol - 1) if pl.cuda_native() else lane // ng",
                                    "jnp.minimum(lane // ng, ncol - 1)")
    normalized = normalized.replace("and pl_cuda_native():", 'and jax.default_backend() != "cpu":')
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    rows.append({"path": rel, "original_sha256": sha(original), "candidate_sha256": sha(candidate),
                 "cuda_normalized_sha256": sha(normalized), "cuda_normalized_byte_equal": normalized == original,
                 "source_lines_equal": len(candidate.splitlines()) == len(original.splitlines()),
                 "pallas_call_sites": candidate.count("pallas_call(")})
result = {"base": base, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "scope": "static CUDA-arm source and line identity; no GPU execution",
          "files": rows, "pass": all(r["cuda_normalized_byte_equal"] and r["source_lines_equal"] for r in rows)}
output.write_text(json.dumps(result, indent=2) + "\n")
assert result["pass"], [r["path"] for r in rows if not r["cuda_normalized_byte_equal"]]
print(json.dumps({"pass": result["pass"], "files": len(rows), "calls": sum(r["pallas_call_sites"] for r in rows)}))
