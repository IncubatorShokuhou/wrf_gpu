"""w44-rc v0.4.4: IDENTITY_BRANCH.json from the schema-2 OFF-identity compares (+ the canonical release_identity shas).

usage: python3 -I identity_branch.py OUT_DIR PIN RC_SWISS RC_PROD_NESTED RC_PRODTOPO0_NESTED SWISS_V043 PROD_V043
RC_* = off_identity_compare.py exit codes (0 IDENTICAL, 1 DIFFERENT, 2 INVALID_INPUT = BLIND/ERROR/not comparable).
Swiss branch (pre-registered, PREREG rules.swiss_identity, FINDINGS E222): 0 -> a_identical_to_v043, 1 -> b_changed_program,
anything else -> INVALID_STOP (no branch; the release does not proceed on an unproven identity).
Expectations recorded, never used to pass anything: PROD nested DIFFERENT (topo fix), PROD topo_shading=0 nested IDENTICAL.
"""
import json
import sys
from pathlib import Path

out, pin = Path(sys.argv[1]), sys.argv[2]
rc_swiss, rc_prod, rc_topo0 = (int(x) if x.lstrip("-").isdigit() else 2 for x in sys.argv[3:6])
swiss_old, prod_old = sys.argv[6], sys.argv[7]
V = {0: "IDENTICAL", 1: "DIFFERENT"}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def relid(arm):
    r = load(out / f"release_identity_{arm}.json") or {}
    return ((r.get("arms") or {}).get("release") or {}).get("sha")


cmp = {c: (load(out / f"CMP_{c}.json") or {}) for c in ("swiss", "prod_nested", "prodtopo0_nested")}
schema2 = {c: dict(verdict=V.get(rc, "INVALID_INPUT"), exit=rc, detail=cmp[c].get("detail"), hint=cmp[c].get("hint"),
                   measured=cmp[c].get("measured"))
           for c, rc in (("swiss", rc_swiss), ("prod_nested", rc_prod), ("prodtopo0_nested", rc_topo0))}
branch = {0: "a_identical_to_v043", 1: "b_changed_program"}.get(rc_swiss, "INVALID_STOP")
s, p = relid("swiss"), relid("prod")
expect = {"prod_nested": "DIFFERENT", "prodtopo0_nested": "DIFFERENT"}
mismatch = [f"{c}: expected {e}, measured {schema2[c]['verdict']}" for c, e in expect.items() if schema2[c]["verdict"] != e]
res = dict(pin=pin, convention="schema-2 OFF probe (o5-port 446807db75 copy) decides; release_identity.py = second convention",
           swiss_branch=branch, schema2=schema2, expectations=expect, expectation_mismatches=mismatch,
           canonical_release_identity=dict(swiss_sha=s, prod_sha=p, swiss_v043=swiss_old, prod_v043=prod_old,
                                           swiss_equal_v043=(s == swiss_old) if s else None,
                                           prod_equal_v043=(p == prod_old) if p else None),
           swiss_sha=s, prod_sha=p, prod_changed=(schema2["prod_nested"]["verdict"] == "DIFFERENT"),
           convention_disagreement=(s is not None and (s == swiss_old) != (rc_swiss == 0)),
           rule="PREREG rules.swiss_identity: (a) C1/C2/C3 vs v0.4.3 must be identical; (b) B9_swiss_24h D6 vs RD11/cpu; "
                "INVALID_STOP -> no GPU release step until the identity is measured validly")
(out / "IDENTITY_BRANCH.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("swiss_branch", "expectation_mismatches", "convention_disagreement")}
                 | {c: v["verdict"] for c, v in schema2.items()}))
