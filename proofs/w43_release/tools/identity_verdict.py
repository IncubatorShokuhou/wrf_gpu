"""RELEASE_PLAN.md section 1 decision: compiled-numerics identity v0.4.2 bundle -> v0.4.3 bundle (CPU, reads receipts).

usage: python3 identity_verdict.py FOLLOWUP_DIR AOTKEY_COMPONENTS_JSON PIN_RECEIPT_DIR OUT_JSON
Inputs: PIN_RECEIPT_DIR/release_identity_{swiss,prod}.json (C0a), AOTKEY COMPONENTS.json (C0b), FOLLOWUP_DIR/
NATIVE_IDENTITY.json (C1/C2/C2c), AUTOTUNE_SEED_CHECK.json (C2b, disclosed), BYTES_VS_V042.json (C3/C4).
Decision (pre-registered): C0a+C0b+C1/C2+C3 PASS -> IDENTICAL (v0.4.2 numerical validation carries forward);
C0 PASS, C2 FAIL -> SEPARATE_COMPILE (D6_REQUIRED, no carry-forward claim); C2 PASS, C3 FAIL -> STOP_DIAGNOSE;
anything missing -> INCOMPLETE.
"""
import json, sys
from pathlib import Path

fu, comp, pin, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


EXPECT = {"swiss": "b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a",
          "prod": "321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b"}
c0a = {}
for arm, sha in EXPECT.items():
    txt = (pin / f"release_identity_{arm}.json").read_text() if (pin / f"release_identity_{arm}.json").exists() else ""
    c0a[arm] = "PASS" if sha in txt else ("MISSING" if not txt else "FAIL")
checks = {"C0a_release_identity": "PASS" if all(v == "PASS" for v in c0a.values()) else
          ("INCOMPLETE" if "MISSING" in c0a.values() else "FAIL"),
          "C0b_aotkey_components": (load(comp) or {}).get("verdict", "INCOMPLETE"),
          "C1_C2_native_identity": (load(fu / "NATIVE_IDENTITY.json") or {}).get("verdict", "INCOMPLETE"),
          "C2b_autotune_seed_unchanged (disclosed)": (load(fu / "AUTOTUNE_SEED_CHECK.json") or {}).get("verdict", "INCOMPLETE"),
          "C3_C4_bytes_vs_v042": (load(fu / "BYTES_VS_V042.json") or {}).get("verdict", "INCOMPLETE")}
c0 = [checks["C0a_release_identity"], checks["C0b_aotkey_components"]]
c2 = checks["C1_C2_native_identity"]
pairs = ((load(fu / "BYTES_VS_V042.json") or {}).get("pairs") or {})
c3 = ("INCOMPLETE" if not pairs or any(v.get("verdict") == "MISSING" for v in pairs.values())
      else ("PASS" if all(v.get("verdict") == "PASS" for v in pairs.values()) else "FAIL"))
if any(v != "PASS" for v in c0):
    decision = "FAIL" if "FAIL" in c0 or "FAIL_OR_MISSING" in c0 else "INCOMPLETE"
elif c2 not in ("PASS", "FAIL") or c3 == "INCOMPLETE":
    decision = "INCOMPLETE"
elif c2 == "PASS":
    decision = "IDENTICAL" if c3 == "PASS" else "STOP_DIAGNOSE"
else:
    decision = "SEPARATE_COMPILE"
res = {"checks": checks, "c0a_detail": c0a, "decision": decision,
       "consequence": {"IDENTICAL": "v0.4.2 numerical validation carries forward (RW3 D6 72 h, icloud=0 D6, teacher tables); RW3 adds timing only",
                       "SEPARATE_COMPILE": "D6_REQUIRED: run RW3 D6 72 h + integrity on v0.4.3; no 'v0.4.2 numerics = v0.4.3' claim",
                       "STOP_DIAGNOSE": "native equal but outputs differ: stop and diagnose before release",
                       "INCOMPLETE": "evidence missing: no claim", "FAIL": "C0 failed: traced program changed"}[decision]}
out.write_text(json.dumps(res, indent=1) + "\n")
print("IDENTITY", decision, checks)
