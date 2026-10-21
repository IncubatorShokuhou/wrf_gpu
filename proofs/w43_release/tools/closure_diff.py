"""C0b/C0c helper: the cheap-key source_fingerprint closure (aot_cheap_key._trace_reachable_source_files) of two trees,
per-file SHA-256 diff + both source_fingerprint_hash values + fn_identity of _advance_chunk_fori with the version string
as shipped and counterfactually reverted (CPU only, imports gpuwrf in a child per tree, no case load).

usage: JAX_PLATFORMS=cpu python3 closure_diff.py TREE_A TREE_B OUT_JSON
"""
import hashlib, json, subprocess, sys
from pathlib import Path

CHILD = r'''
import hashlib, json, sys
from pathlib import Path
import gpuwrf
from gpuwrf.runtime import aot_cheap_key as k
root = Path(gpuwrf.__file__).resolve().parent
files = k._trace_reachable_source_files(root)
out = {"version": gpuwrf.__version__, "pkg_root": str(root),
       "closure": {str(Path(f).resolve().relative_to(root)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (files or [])},
       "closure_ok": files is not None, "source_fingerprint": k.source_fingerprint_hash()}
from gpuwrf.runtime.operational_mode import _advance_chunk_fori
out["fn_identity"] = k.fn_identity_hash(_advance_chunk_fori)
gpuwrf.__version__ = sys.argv[1]
out["fn_identity_with_version_" + sys.argv[1]] = k.fn_identity_hash(_advance_chunk_fori)
print("JSON" + json.dumps(out))
'''
a, b, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def probe(tree, other_version):
    env = {"JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(tree / "src"),
           "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    r = subprocess.run([sys.executable, "-c", CHILD, other_version], env=env, capture_output=True, text=True, timeout=900,
                       cwd=str(tree))
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        raise SystemExit(f"probe failed for {tree}: {r.stderr[-2000:]}")
    return json.loads(line[-1][4:])


A = probe(a, "0.4.3"); B = probe(b, "0.4.2")
ca, cb = A["closure"], B["closure"]
res = {"tree_a": str(a), "tree_b": str(b), "version_a": A["version"], "version_b": B["version"],
       "closure_files": (len(ca), len(cb)), "only_a": sorted(set(ca) - set(cb)), "only_b": sorted(set(cb) - set(ca)),
       "differing_files": sorted(f for f in set(ca) & set(cb) if ca[f] != cb[f]),
       "source_fingerprint": (A["source_fingerprint"], B["source_fingerprint"]),
       "fn_identity": (A["fn_identity"], B["fn_identity"]),
       "fn_identity_b_with_version_of_a": B["fn_identity_with_version_0.4.2"],
       "fn_identity_equal_after_version_revert": B["fn_identity_with_version_0.4.2"] == A["fn_identity"],
       "cli_or_host_suspend_in_closure": [f for f in cb if f in ("cli.py", "runtime/host_suspend.py")]}
outp.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k not in ("tree_a", "tree_b")}, indent=1))
