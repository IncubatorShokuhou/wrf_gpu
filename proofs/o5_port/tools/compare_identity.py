"""OFF identity: base vs candidate capture_real.py receipts (same alias path, CUDA platform, identity convention).

usage: compare_identity.py BASE_DIR CAND_DIR OUT.json
Checks canonical jaxpr sha (release identity), raw CUDA-target HLO sha/bytes, and the per-kernel Triton IR
both location-stripped and WITH locations (E109: line shifts show up there), plus the raw TTIR files bytewise.
"""
import hashlib
import json
import sys
from pathlib import Path

base, cand, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rb, rc = (json.loads((d / "receipt.json").read_text()) for d in (base, cand))
res = {"base": str(base), "cand": str(cand), "checks": {}}
for key in ("canon_sha256", "raw_hlo_sha256", "raw_hlo_bytes", "native_custom_calls", "triton_custom_calls",
            "n_kernels_unique", "n_kernel_lowerings", "grid"):
    res["checks"][key] = {"base": rb.get(key), "cand": rc.get(key), "equal": rb.get(key) == rc.get(key)}


LAUNCH = "--ignore-launch" not in sys.argv  # P01 (first tool) lacked num_warps; raw HLO equality covers them


def kset(r, field):
    return sorted((k[field], k["num_warps"], k["num_stages"], k["calls"]) if LAUNCH else (k[field], k["calls"])
                  for k in r["kernels"] if field in k)


for field in ("ttir_sha16", "ttir_loc_sha16"):
    res["checks"][f"kernels_{field}"] = {"equal": kset(rb, field) == kset(rc, field),
                                         "only_base": sorted(set(kset(rb, field)) - set(kset(rc, field)))[:10],
                                         "only_cand": sorted(set(kset(rc, field)) - set(kset(rb, field)))[:10]}
files_b = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (base / "ttir").glob("*.ttir")}
files_c = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cand / "ttir").glob("*.ttir")}
res["checks"]["ttir_files_bytewise"] = {"equal": files_b == files_c, "n": len(files_b)}
raw_b, raw_c = (d / "raw.hlo" for d in (base, cand))
if raw_b.exists() and raw_c.exists():
    res["checks"]["raw_hlo_file_bytes_equal"] = {"equal": raw_b.read_bytes() == raw_c.read_bytes()}
res["verdict"] = "IDENTICAL" if all(v["equal"] for v in res["checks"].values()) else "DIFFERENT"
outp.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v["equal"] for k, v in res["checks"].items()}), res["verdict"])
