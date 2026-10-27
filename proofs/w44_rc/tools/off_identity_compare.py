#!/usr/bin/env python3
"""Compare two off_identity_probe.py receipts (base vs candidate, schema 2).

usage: off_identity_compare.py BASE.json CAND.json [OUT.json]
exit 0 IDENTICAL, 1 DIFFERENT, 2 INVALID_INPUT (a BLIND/ERROR receipt, or receipts that did not measure the
same case / programs / toolchain / release-default state / JAX_*, XLA_*, GPUWRF_* environment).
IDENTICAL requires, per covered program, the full-constant HLO hash (constant VALUES included), the legacy raw
HLO hash, the Pallas counts per kernel name and every kernel body (with and without source lines), plus the
canonical release jaxpr and its constant values.
"""
import json
import sys

SAME_INPUT = (("case",), ("covered_programs",), ("max_dom",), ("release_defaults_active",), ("operator_env",),
              ("provenance", "versions"), ("provenance", "program_env"), ("canon_traced",))
PROGRAM_KEYS = ("raw_hlo_full_sha256", "raw_hlo_sha256", "triton_custom_calls", "n_kernel_bodies",
                "n_pallas_lowerings", "lowerings_by_name")


def _get(rec, path):
    for key in path:
        rec = rec.get(key) if isinstance(rec, dict) else None
    return rec


def compare(base, cand):
    res = {"measured": {side: {k: _get(r, ("provenance", k)) for k in ("realpath", "git_head", "git_src_dirty",
                                                                     "src_gpuwrf_content_sha256")}
                        for side, r in (("base", base), ("cand", cand))},
           "covered_programs": base.get("covered_programs"), "checks": {}}
    if base.get("schema") != 2 or cand.get("schema") != 2:
        return dict(res, verdict="INVALID_INPUT", detail="schema-2 receipts required (full-constant hashes)")
    if base.get("verdict") != "COMPLETE" or cand.get("verdict") != "COMPLETE":
        return dict(res, verdict="INVALID_INPUT",
                    detail=f"verdicts {base.get('verdict')}/{cand.get('verdict')}: a BLIND or failed probe proves nothing")
    for rec in (base, cand):  # D3: whether the canonical pass ran is part of what was measured
        rec.setdefault("canon_traced", "canon_sha256" in rec)
    if base.get("order", "native-first") != "native-first" or cand.get("order", "native-first") != "native-first":
        return dict(res, verdict="INVALID_INPUT", detail="only native-first receipts prove identity (D4)")
    mismatch = [".".join(p) for p in SAME_INPUT if _get(base, p) != _get(cand, p)]
    if mismatch:
        return dict(res, verdict="INVALID_INPUT", detail=f"receipts did not measure the same thing: {mismatch}")
    for key in ("canon_sha256", "canon_consts_sha256"):
        res["checks"][key] = {"base": base.get(key), "cand": cand.get(key), "equal": base.get(key) == cand.get(key)}
    for program in base["covered_programs"]:
        pb, pc = base["programs"][program], cand["programs"][program]
        for key in PROGRAM_KEYS:
            res["checks"][f"{program}.{key}"] = {"equal": pb.get(key) == pc.get(key),
                                                  **({} if key == "lowerings_by_name" else
                                                     {"base": pb.get(key), "cand": pc.get(key)})}
    for field in ("ttir_sha16", "ttir_loc_sha16"):
        kb, kc = ({(k["program"], k[field], k["num_warps"], k["num_stages"], k["calls"]) for k in r["kernels"]}
                  for r in (base, cand))
        res["checks"][f"kernels_{field}"] = {"equal": kb == kc, "only_base": sorted(kb - kc)[:10],
                                             "only_cand": sorted(kc - kb)[:10]}
    res["verdict"] = "IDENTICAL" if all(c["equal"] for c in res["checks"].values()) else "DIFFERENT"
    if (res["verdict"] == "DIFFERENT" and res["checks"]["kernels_ttir_sha16"]["equal"]
            and not res["checks"]["kernels_ttir_loc_sha16"]["equal"]):
        res["hint"] = "all kernel bodies equal without source lines: a line shift in a Pallas file (E109/E128)?"
    return res


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base, cand = (json.load(open(p)) for p in argv[:2])
    res = dict(compare(base, cand), base=argv[0], cand=argv[1])
    if len(argv) > 2:
        json.dump(res, open(argv[2], "w"), indent=1)
    print("programs:", res.get("covered_programs"), "| measured:", json.dumps(res["measured"]))
    print(json.dumps({k: v["equal"] for k, v in res["checks"].items()}), res["verdict"], res.get("detail", ""),
          res.get("hint", ""))
    return {"IDENTICAL": 0, "DIFFERENT": 1}.get(res["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
