"""Resolve E41's non-leaf CALLEE rule from actual cuobjdump CALL targets.

Large kernel callers are not themselves callees (FINDINGS E189). Unresolved
control flow is reported UNKNOWN, never silently accepted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+([^/]+)", re.I)
TARGET = re.compile(r"\b(?:CALL|BRA)(?:\.[A-Z]+)*\s+(?:!?U?P(?:\d+|T),\s*)?0x([0-9a-f]+)", re.I)


def parse(text):
    functions, current = {}, None
    for line in text.splitlines():
        head = FUNCTION.match(line)
        if head:
            current = functions.setdefault(head.group(1), {})
            continue
        ins = INSTRUCTION.match(line)
        if current is not None and ins:
            pc, op = int(ins.group(1), 16), ins.group(2).strip()
            if pc in current:
                raise ValueError("duplicate instruction address")
            current[pc] = op
    if not functions or not all(functions.values()):
        raise ValueError("empty function/instruction inventory")
    return functions


def analyze(functions):
    rows, unknown, failures = [], [], []
    for name, instructions in functions.items():
        pcs = sorted(instructions)
        following = dict(zip(pcs, pcs[1:]))
        calls = [(pc, op) for pc, op in instructions.items() if re.search(r"\bCALL\.", op)]
        targets = set()
        for pc, op in calls:
            target = TARGET.search(op)
            if target is None or int(target.group(1), 16) not in instructions:
                unknown.append({"function": name, "pc": pc, "op": op, "reason": "unresolved CALL"})
            else:
                targets.add(int(target.group(1), 16))
        for start in sorted(targets):
            pending, seen, nested, returned = [start], set(), 0, False
            while pending:
                pc = pending.pop()
                if pc in seen:
                    continue
                if pc not in instructions:
                    unknown.append({"function": name, "start": start, "pc": pc,
                                    "reason": "control flow outside disassembly"})
                    continue
                seen.add(pc)
                op = instructions[pc]
                conditional = op.startswith("@") or bool(
                    re.search(r"\bBRA(?:\.[A-Z]+)*\s+!?U?P(?:\d+|T),", op))
                nested += bool(re.search(r"\bCALL\.", op))
                if re.search(r"\b(?:BRX|JMX|JMP)(?:\.|\s)", op):
                    unknown.append({"function": name, "start": start, "pc": pc, "op": op,
                                    "reason": "indirect/unimplemented branch"})
                terminal = bool(re.search(r"\b(?:RET|EXIT)(?:\.|\s|;)", op))
                if terminal:
                    returned = True
                    if not conditional:
                        continue
                branch = bool(re.search(r"\bBRA(?:\.|\s)", op))
                if branch:
                    target = TARGET.search(op)
                    if target is None:
                        unknown.append({"function": name, "start": start, "pc": pc,
                                        "op": op, "reason": "unresolved BRA"})
                    else:
                        pending.append(int(target.group(1), 16))
                    if not conditional:
                        continue
                if pc in following:
                    pending.append(following[pc])
                else:
                    unknown.append({"function": name, "start": start, "pc": pc,
                                    "reason": "callee falls past function end"})
            if not returned:
                unknown.append({"function": name, "start": start, "reason": "no reachable return"})
            row = {"function": name, "target_pc": start, "instructions": len(seen), "nested_calls": nested}
            rows.append(row)
            if nested and len(seen) > 1000:
                failures.append(row)
    return {"gate": "FAIL" if failures else ("UNKNOWN" if unknown else "PASS"),
            "function_count": len(functions), "callee_count": len(rows),
            "max_callee_instructions": max((r["instructions"] for r in rows), default=0),
            "max_nonleaf_callee_instructions": max((r["instructions"] for r in rows if r["nested_calls"]), default=0),
            "failures": failures, "unknown": unknown, "callees": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import gpu_smoke_e41 as fallback
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    gate = fallback._gate_module()
    blob = a.executable.read_bytes()
    fmt = ax._gpu_blob_format(blob)
    if fmt not in (ax.GPU_THUNK_FORMAT, ax.GPU_LEGACY_FORMAT):
        raise ValueError("not a serialized GPU executable")
    native = gate._record(blob, 2, ax, asm) if fmt == ax.GPU_THUNK_FORMAT else blob
    a.out.mkdir(parents=True, exist_ok=False)
    results = []
    for i, cubin in enumerate(gate._elfs(native)):
        path = a.out / f"native_{i}.cubin"
        path.write_bytes(cubin)
        try:
            sass = subprocess.run([fallback.CUOBJDUMP, "-sass", str(path)], check=True,
                                  capture_output=True, text=True, timeout=300).stdout
        finally:
            path.unlink(missing_ok=True)
        (a.out / f"native_{i}.sass.gz").write_bytes(gzip.compress(sass.encode(), mtime=0))
        result = analyze(parse(sass))
        result.update(elf=i, cubin_sha256=hashlib.sha256(cubin).hexdigest(),
                      sass_sha256=hashlib.sha256(sass.encode()).hexdigest())
        results.append(result)
    if not results:
        raise ValueError("no native ELF")
    verdict = "FAIL" if any(r["gate"] == "FAIL" for r in results) else (
        "UNKNOWN" if any(r["gate"] == "UNKNOWN" for r in results) else "PASS")
    receipt = {"gate": verdict, "executable_sha256": hashlib.sha256(blob).hexdigest(), "elfs": results}
    (a.out / "callee_gate.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"gate": verdict, "callee_count": sum(r["callee_count"] for r in results),
                      "unknown": sum(len(r["unknown"]) for r in results),
                      "max_nonleaf_callee_instructions": max(r["max_nonleaf_callee_instructions"] for r in results)}))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
