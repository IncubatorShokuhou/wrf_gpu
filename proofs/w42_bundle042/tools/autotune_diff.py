"""w42-bundle042: compare the per-fusion XLA autotune decisions of two sealed bundles (CPU, reads files only).

usage: python3 autotune_diff.py BUNDLE_A BUNDLE_B OUT_JSON [VARDIFF_JSON]
Each entry of <bundle>/jax/xla_gpu_per_fusion_autotune_cache_dir is keyed by the fusion fingerprint (fusion HLO + device);
its content is the chosen kernel (emitter + config). Same fingerprint with different content = the same fusion compiled to
a different kernel by a separate autotune run (E99 mechanism). Reports counts per (A choice -> B choice) emitter pair and
a few examples; with VARDIFF_JSON (var_diff.py N vs O) it records which wrfout files/variables differ next to it.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

A, B, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
VD = Path(sys.argv[4]) if len(sys.argv) > 4 else None
sub = "jax/xla_gpu_per_fusion_autotune_cache_dir"


def entries(root):
    return {p.name: p.read_bytes() for p in sorted((root / sub).iterdir()) if p.is_file()}


def emitter(b):
    m = re.search(rb'name: "([A-Z_]+)"', b)
    return m.group(1).decode() if m else "?"


ea, eb = entries(A), entries(B)
common = sorted(set(ea) & set(eb))
diff = [n for n in common if ea[n] != eb[n]]
pairs = collections.Counter(f"{emitter(ea[n])} -> {emitter(eb[n])}" for n in diff)
res = dict(
    bundle_a=str(A), bundle_b=str(B), dir=sub,
    a_entries=len(ea), b_entries=len(eb), common_fingerprints=len(common), identical_choice=len(common) - len(diff),
    different_choice=len(diff), only_a=sorted(set(ea) - set(eb)), only_b=sorted(set(eb) - set(ea)),
    different_choice_by_emitter=dict(pairs.most_common()),
    examples=[dict(fingerprint=n, a=ea[n].decode(errors="replace")[:400], b=eb[n].decode(errors="replace")[:400])
              for n in diff[:5]],
    a_dir_sha256=hashlib.sha256(b"".join(n.encode() + ea[n] for n in sorted(ea))).hexdigest(),
    b_dir_sha256=hashlib.sha256(b"".join(n.encode() + eb[n] for n in sorted(eb))).hexdigest())
if VD:
    v = json.loads(VD.read_text())
    res["vardiff"] = dict(file=str(VD), sha256=hashlib.sha256(VD.read_bytes()).hexdigest(), common_files=v["common_files"],
                          differing_variables=len(v["only_vars_differ"]),
                          global_attrs_differ=sorted(v["global_attr_diff"]), structure_diff=v["structure_diff"])
OUT.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("a_entries", "b_entries", "common_fingerprints", "identical_choice",
                                      "different_choice", "different_choice_by_emitter")} | {"only_a": len(res["only_a"]),
                                                                                          "only_b": len(res["only_b"])}))
