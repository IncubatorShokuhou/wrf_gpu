"""B1 nested-program GPU gate, key part (manager 2026-10-18T10:43:12Z): ONE fused cheap key across 0227/0502/0614.

usage: python fusedkey.py ROOT        (ROOT = B1 root of the pin; reads JSON only, no JAX, no ALISIOS path)
The pinned fix makes the fused d02/d03 cascade take its namelists as jit arguments, so its cheap key must not depend on
the case date. Evidence per WN3 launcher arm (cold3, warm3, par24) and case, from the run's own proof
wrfout/proofs/nested_pipeline_run.json (metadata.nested_aot.domains, written by the measured process): cheap_key,
hlo_sha256, blob_sha256, source, loaded, aot_written of every stepping program; plus the launcher receipt's
aot_cold_executables and the cache_wn3 AOT inventory (one .meta/.xlaexec set per key and program).
Verdict ONE_KEY only if all three par24 cases record the same fused cheap key, equal to the cold3 capture, and the cache
holds no other fused key; a missing proof/case is INCOMPLETE, never ONE_KEY. Writes ROOT/FUSED_KEYS.json.
"""
import json, sys
from pathlib import Path

FIELDS = ("cheap_key", "hlo_sha256", "blob_sha256", "source", "loaded", "aot_written", "aot_error", "cache_hit_count",
          "load_count")
CASES = ("wn3_0227", "wn3_0502", "wn3_0614")


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


root = Path(sys.argv[1]).resolve()
out = {"root": str(root), "arms": {}, "cache": {}}
for arm in ("cold3", "warm3", "par24"):
    for case in CASES:
        proof = load(root / "wn3_runs" / arm / case / "wrfout/proofs/nested_pipeline_run.json")
        if proof is None:
            continue
        programs = ((proof.get("metadata") or {}).get("nested_aot") or {}).get("domains") or {}
        receipt = load(root / "wn3_runs" / arm / case / "receipt.json") or {}
        out["arms"].setdefault(arm, {})[case] = {
            "programs": {name: {k: rec.get(k) for k in FIELDS} for name, rec in programs.items() if isinstance(rec, dict)},
            "aot_cold_executables": receipt.get("aot_cold_executables"), "aot_cache_warm": receipt.get("aot_cache_warm")}
for tag in sorted((root / "cache_wn3/jax/aot").glob("*")):
    for prog in sorted(p for p in tag.iterdir() if p.is_dir()):
        metas = {}
        for meta in sorted(prog.glob("*.meta")):
            body = load(meta)
            metas[meta.name] = {"hlo_sha256": body.get("hlo_sha256") if isinstance(body, dict) else None,
                                "xlaexec": meta.with_suffix(".xlaexec").is_file(),
                                "inode": meta.with_suffix(".xlaexec").stat().st_ino if meta.with_suffix(".xlaexec").is_file() else None}
        out["cache"][f"{tag.name}/{prog.name}"] = metas

fused = lambda arm, case: next((rec.get("cheap_key") for name, rec in  # noqa: E731
                                 ((out["arms"].get(arm) or {}).get(case) or {}).get("programs", {}).items()
                                 if name.startswith("fused")), None)
par = {case: fused("par24", case) for case in CASES}
cold = fused("cold3", "wn3_0227")
cache_fused = sorted({name for prog, metas in out["cache"].items() if prog.split("/")[-1].startswith("fused")
                      for name in metas if name.startswith("k_")})
out["fused_key"] = {"par24": par, "cold3_0227": cold, "cache_fused_cheap_key_metas": cache_fused}
if not all(par.values()) or cold is None:
    out["verdict"] = "INCOMPLETE"
elif len(set(par.values())) == 1 and par["wn3_0227"] == cold and len(cache_fused) <= 1:
    out["verdict"] = "ONE_KEY"
else:
    out["verdict"] = "KEY_SPLIT"
(root / "FUSED_KEYS.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"verdict": out["verdict"], "fused_key": out["fused_key"]}, indent=1))
