"""Independent positive, hard-negative and unknown controls for the SASS rule."""
import json

from callee_gate import analyze, parse


text = """Function : test
 /*0000*/ CALL.REL.NOINC 0x40; /* encoding */
 /*0010*/ EXIT; /* encoding */
 /*0040*/ MOV R0, R1; /* encoding */
 /*0050*/ RET.REL.NODEC R20 0x0; /* encoding */
"""
result = analyze(parse(text))
assert result["gate"] == "PASS" and result["max_callee_instructions"] == 2

nonleaf = {0: "CALL.REL.NOINC 0x100;", 16: "EXIT;"}
nonleaf.update({256 + 16 * n: "NOP;" for n in range(1050)})
nonleaf[256 + 16 * 1000] = "CALL.REL.NOINC 0x10000;"
nonleaf[256 + 16 * 1050] = "RET.REL.NODEC R20 0x0;"
nonleaf[65536] = "RET.REL.NODEC R20 0x0;"
result = analyze({"test": nonleaf})
assert result["gate"] == "FAIL" and result["failures"][0]["instructions"] == 1051

leaf = dict(nonleaf)
leaf[256 + 16 * 1000] = "NOP;"
result = analyze({"test": leaf})
assert result["gate"] == "PASS" and result["max_callee_instructions"] == 1051

result = analyze({"test": {0: "CALL.REL.NOINC R0;", 16: "EXIT;"}})
assert result["gate"] == "UNKNOWN"

result = analyze({"test": {0: "CALL.REL.NOINC 0x40;", 16: "EXIT;", 64: "@P0 BRA 0x70;",
                            80: "NOP;", 96: "RET.REL.NODEC R20 0x0;", 112: "RET.REL.NODEC R20 0x0;"}})
assert result["gate"] == "PASS" and result["max_callee_instructions"] == 4
result = analyze({"test": {0: "CALL.REL.NOINC 0x40;", 16: "EXIT;", 64: "BRA !P4, 0x70;",
                            80: "NOP;", 96: "RET.REL.NODEC R20 0x0;", 112: "RET.REL.NODEC R20 0x0;"}})
assert result["gate"] == "PASS" and result["max_callee_instructions"] == 4

caller = {16 * n: "NOP;" for n in range(1200)}
caller.update({19200: "CALL.REL.NOINC 0x8000;", 19216: "EXIT;",
               32768: "NOP;", 32784: "RET.REL.NODEC R20 0x0;"})
result = analyze({"large_caller": caller})
assert result["gate"] == "PASS" and result["max_callee_instructions"] == 2
print(json.dumps({"pass_cases": 7, "scope": "scanner rule controls only, no model fidelity claim"}))
