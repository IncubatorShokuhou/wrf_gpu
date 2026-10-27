#!/usr/bin/env python3
"""E41 verdict from <E41_DIR>/wave_gate.json -- never from wave_cubin_gate's exit code (FINDINGS E230).

wave_cubin_gate exits 2 for FAIL *and* INCOMPLETE; INCOMPLETE is the normal verdict whenever compact AOT
programs are scanned (E201: their kInput HLO is UNKNOWN_AOT_METADATA). Acceptable here = PASS, or
INCOMPLETE whose only unknowns are that compact-AOT metadata, with STACK PASS, no non-leaf callee or kInput
flag and no unknown native call graph (the rule cr-w44-launchers accepted for w44-topo/w44-warmstart v8).
Everything else (FAIL, another UNKNOWN status, unknown call graph, missing file) is recorded as not
acceptable. The launcher records the verdict and continues; the lane's kill switch acts on it afterwards.

usage: e41_verdict.py E41_DIR OUT.json   exit 0 = verdict written (any verdict), 1 = inconsistent receipt
"""
import hashlib
import json
import sys
from pathlib import Path


def verdict(e41_dir):
    path = Path(e41_dir) / "wave_gate.json"
    if not path.is_file():
        return {"path": str(path), "e41_gate": "MISSING", "acceptable": False, "reason": "no wave_gate.json"}
    gate = json.loads(path.read_text())
    e41, hard, flags = gate["e41_gate"], gate["hard_stack_gate"], bool(gate["conservative_flags"])
    assert e41 in ("PASS", "INCOMPLETE", "FAIL") and hard in ("PASS", "FAIL"), (e41, hard)
    assert (e41 == "FAIL") == (hard == "FAIL" or flags), (e41, hard, flags)
    entries = gate["entries"]
    status = sorted({str(r.get("kinput_transpose_status", "")) for r in entries})
    unknown = [s for s in status if s.startswith("UNKNOWN")]
    unknown_native = sorted({r["entry"] for r in entries for elf in r["elfs"] if elf.get("callgraph_unknown")})
    assert (e41 == "INCOMPLETE") == (not flags and hard == "PASS" and bool(unknown or unknown_native)), (e41, unknown)
    if e41 == "PASS":
        acceptable, reason = True, "PASS"
    elif e41 == "INCOMPLETE" and unknown == ["UNKNOWN_AOT_METADATA"] and not unknown_native:
        acceptable, reason = True, "INCOMPLETE: STACK PASS, no callee/kInput flag, only compact-AOT kInput metadata unknown"
    else:
        acceptable, reason = False, f"{e41}: hard {hard}, flags {flags}, unknown {unknown}, unknown native {unknown_native}"
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "e41_gate": e41,
            "hard_stack_gate": hard, "conservative_flags": flags, "n_entries": gate["n_entries"],
            "aot_programs": gate["coverage"]["aot_programs"], "kinput_status": status,
            "unknown_native_entries": unknown_native,
            "max_stack_B": max((e["max_stack_B"] for r in entries for e in r["elfs"]), default=None),
            "acceptable": acceptable, "reason": reason}


def main(argv):
    e41_dir, out = argv
    try:
        record = verdict(e41_dir)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        print("E41 VERDICT UNREADABLE", type(error).__name__, error)
        return 1
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    print("E41 VERDICT", record["e41_gate"], "acceptable" if record["acceptable"] else "NOT acceptable", "|", record["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
