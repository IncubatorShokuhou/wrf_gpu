"""Merge the per-batch caller-site reach parts into proofs/final_rerun/AFFECTED.json.

usage: python merge_affected.py <parts_dir> <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path

MAIN = "04cb09f816d1284e72c515eca64a94909f40e071"


def _norm(text):
    t = text.strip().lower()
    for key, word in (("yes", "yes"), ("byte", "BYTE"), ("history", "history-only"), ("refusal", "refusal-only"),
                      ("restart", "restart-only"), ("none", "none")):
        if t.startswith(key):
            return word
    raise ValueError(f"unclassified numeric_change: {text!r}")


def main():
    parts_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    comps = json.loads((parts_dir / "composites.json").read_text())
    cases = {c["id"]: c for c in comps["cases"]}
    act = json.loads((parts_dir / "active.json").read_text())
    active = set(act["active"])
    fixes, part_sha = [], {}
    for name in ("part_A.json", "part_B.json", "part_C.json", "part_D.json"):
        raw = (parts_dir / name).read_bytes()
        part_sha[name] = hashlib.sha256(raw).hexdigest()
        for fix in json.loads(raw):
            fix["numeric_change_detail"] = fix["numeric_change"]
            fix["numeric_change"] = _norm(fix["numeric_change"])
            fixes.append(fix)
    unknown = sorted({cid for f in fixes for cid in f["affected"] if cid not in cases})
    assert not unknown, unknown
    per = {}
    for cid, case in cases.items():
        reach = [f for f in fixes if cid in f["affected"]]
        numeric = [f for f in reach if f["numeric_change"] == "yes"]
        runnable = cid in active
        per[cid] = dict(status=case["status"], runnable=runnable, options=case["options"], extra=case["extra"],
                        features=case["features"],
                        numeric_fixes=[f"b{f['batch']}:{f['fix']}" for f in numeric],
                        class_a_numeric=[f"b{f['batch']}:{f['fix']}" for f in numeric if f["class"] == "A"],
                        other_reach=[f"b{f['batch']}:{f['fix']} ({f['numeric_change']})" for f in reach
                                     if f["numeric_change"] != "yes"])
    rerun = [cid for cid, p in per.items() if p["runnable"] and p["numeric_fixes"]]
    rerun.sort(key=lambda cid: (-len(per[cid]["class_a_numeric"]), -len(per[cid]["numeric_fixes"]), cid))
    payload = dict(
        schema="final-rerun-affected-v1", lane="t-calleraudit", main=MAIN,
        method=("per merged fix of batches 38-56 (src diff per batch on main first-parent), the dispatch/caller "
                "condition at main 04cb09f816 that executes the changed code, evaluated on every composite's "
                "options/extra/features (v-prep MATRIX.json); numeric_change yes|BYTE|history-only|refusal-only|"
                "restart-only; only 'yes' puts a composite on the rerun list"),
        runnable_source=f"{act['source']} active composites at {act['utc']} ({len(active)})",
        composites_source=comps["source"], composites_source_commit=comps["source_commit"], parts_sha256=part_sha,
        fixes=fixes,
        rerun_numeric=rerun,
        rerun_numeric_class_a=[cid for cid in rerun if per[cid]["class_a_numeric"]],
        reached_only_non_numeric=sorted(cid for cid, p in per.items()
                                        if p["runnable"] and not p["numeric_fixes"] and p["other_reach"]),
        not_reached=sorted(cid for cid, p in per.items() if p["runnable"] and not p["numeric_fixes"]
                           and not p["other_reach"]),
        not_runnable=sorted(cid for cid, p in per.items() if not p["runnable"]),
        caveats=[
            "Reach covers the src changes of batches 38-56 only (main first-parent heads 668f27f694..04cb09f816). "
            "A composite outside rerun_numeric may reuse a paid result only if that result's source already "
            "contains every pre-batch-38 change it reaches (v-prep checks its source commit).",
            "BYTE/history-only/refusal-only items are listed per composite under other_reach; they need inventory "
            "(history fields) or E41/timing on the rerun, not a fidelity rerun.",
            "c25 (bl3+sf91) is not runnable: pairing-refusal control, and since batch53 also refused by "
            "'bl3 requires sf3'. c26 (QNSE) reruns for b49 (QNSE P_PHY/P8W); b54 sea-ice admission is refusal-only.",
            "[I] items: F23 SSiB/RUC topographic SW BYTE at slope_rad=0 (c33 traced, c40 producer not traced); "
            "F18 r1 reach for ra4/ra1 not traced (c04 c05 c07 c13 c17 are on the list through F7/F9 anyway); "
            "cu1+mp8 F11 BYTE (c36 c41 c46 c71-73 c90) rests on the all-flags branch + PROD 6487 identity.",
        ],
        manager_mandated_extra=["c87", "c88", "c89", "OVN-YSU3 (c02 c40 c56)", "OVN-CAM89 (c89)"],
        per_composite=per)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("fixes", len(fixes), "rerun", len(rerun), "classA", len(payload["rerun_numeric_class_a"]),
          "non-numeric-only", len(payload["reached_only_non_numeric"]), "not reached", len(payload["not_reached"]),
          "not runnable", len(payload["not_runnable"]))


if __name__ == "__main__":
    main()
