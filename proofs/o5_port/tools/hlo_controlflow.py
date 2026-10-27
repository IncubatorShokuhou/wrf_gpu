"""Static control-flow census of an optimized XLA GPU HLO module (lane o5-port, CPU only).

For one executable (one production step) it reports, weighted by nesting multiplicity
(while: known_trip_count, unknown -> 1 and flagged; conditional: worst branch):
  * launch-like ops per execution (fusion, custom-call, copy, other non-free ops),
  * conditional executions and unknown-trip while iterations = host round trips when control flow is
    NOT captured into command buffers (ROCm today: no conditional/while graph nodes),
  * which while/conditional instructions sit inside command-buffer computations (CUDA +CONDITIONAL,+WHILE).
Static and approximate: not a trace; thunks != HLO ops one-to-one. Labels [I] in the report.

usage: hlo_controlflow.py OPTIMIZED.hlo[.gz] OUT.json
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path, outp = sys.argv[1], sys.argv[2]
text = (gzip.open(path, "rt") if path.endswith(".gz") else open(path)).read()

HEAD = re.compile(r"^(ENTRY\s+)?%?([\w.\-$]+)\s*\(.*\)\s*->\s*.*\{\s*$")
INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-$]+)\s*=\s*(.*)$")
FREE = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "after-all", "partition-id",
        "replica-id", "opt-barrier", "add-dependency", "call", "while", "conditional", "iota"}
comps, entry, cur = {}, None, None
for line in text.splitlines():
    m = HEAD.match(line)
    if m and not line.lstrip().startswith(("ROOT", "%param")) and "=" not in line.split("(")[0]:
        cur = m.group(2)
        comps[cur] = []
        if m.group(1):
            entry = cur
        continue
    if line.strip() == "}":
        cur = None
        continue
    if cur is not None:
        im = INSTR.match(line)
        if im:
            comps[cur].append(im.group(2))

fused = set()
for body in comps.values():
    for ins in body:
        for c in re.findall(r"calls=%?([\w.\-$]+)", ins):
            fused.add(c)


def opcode(ins):
    # "f32[..]{..} fusion(%a, ...), kind=..." -> fusion ; tuple shapes contain spaces/parens
    m = re.search(r"\s([a-z][a-z0-9\-]*)\(", " " + ins)
    return m.group(1) if m else "?"


def trip(ins):
    m = re.search(r'"known_trip_count":\{"n":"(\d+)"\}', ins)
    return int(m.group(1)) if m else None


cache = {}


def walk(name, stack=()):
    """Returns dict of weighted counts for ONE execution of computation `name`."""
    if name in cache:
        return cache[name]
    if name in stack or name not in comps:
        return defaultdict(float)
    acc = defaultdict(float)
    for ins in comps[name]:
        op = opcode(ins)
        if op == "while":
            body = re.search(r"body=%?([\w.\-$]+)", ins).group(1)
            cond = re.search(r"condition=%?([\w.\-$]+)", ins).group(1)
            n = trip(ins)
            acc["while"] += 1
            if n is None:
                acc["while_unknown_trip"] += 1
            reps = n if n is not None else 1
            b, c = walk(body, stack + (name,)), walk(cond, stack + (name,))
            for k, v in b.items():
                acc[k] += v * reps
            for k, v in c.items():
                acc[k] += v * (reps + 1)
            acc["while_iterations"] += reps
            if n is None:
                acc["host_sync_while_iters"] += reps + 1  # predicate DtoH per evaluation
        elif op == "conditional":
            branches = re.findall(r"(?:branch_computations=\{([^}]*)\})|(?:true_computation=%?([\w.\-$]+)), "
                                  r"false_computation=%?([\w.\-$]+)", ins)
            names = []
            for b in branches:
                if b[0]:
                    names += [x.strip().lstrip("%") for x in b[0].split(",")]
                else:
                    names += [b[1], b[2]]
            acc["conditional"] += 1
            acc["host_sync_conditional"] += 1
            sub = [walk(x, stack + (name,)) for x in names]
            keys = set().union(*[s.keys() for s in sub]) if sub else set()
            for k in keys:
                acc[k] += max(s.get(k, 0) for s in sub)
        elif op == "call":
            tgt = re.search(r"to_apply=%?([\w.\-$]+)", ins)
            if tgt:
                sub = walk(tgt.group(1), stack + (name,))
                cb = "command_buffer" in tgt.group(1) or "command_buffer" in ins
                for k, v in sub.items():
                    acc[k] += v
                    if cb and k in ("while", "conditional", "launch_ops"):
                        acc[f"in_command_buffer_{k}"] += v
                if cb:
                    acc["command_buffer_calls"] += 1
        elif op not in FREE:
            acc["launch_ops"] += 1
            acc[f"op_{op}"] += 1
            if op == "custom-call" and "__gpu$xla.gpu.triton" in ins:
                acc["triton_calls"] += 1
    cache[name] = acc
    return acc


res = walk(entry)
summary = {"entry": entry, "computations": len(comps), "fused_computations": len(fused),
           "weighted_per_execution": {k: round(v, 1) for k, v in sorted(res.items()) if not k.startswith("op_")},
           "top_ops": dict(sorted(((k[3:], round(v)) for k, v in res.items() if k.startswith("op_")),
                                  key=lambda kv: -kv[1])[:12]),
           "static_counts": {"while": text.count(" while("), "conditional": text.count(" conditional("),
                             "known_trip_count": text.count("known_trip_count")}}
json.dump(summary, open(outp, "w"), indent=1)
print(json.dumps(summary, indent=1))
