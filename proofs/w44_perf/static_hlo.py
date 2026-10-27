"""Source-attributed static fusion/copy inventory of the same-target CPU screens.

Counts are across all compiled branches, unweighted by runtime/cadence. They
are not launched kernels/step, traffic bytes, GPU duration or a speed verdict.
Metadata and frame tables are from the SAME printed HLO (E82/E182).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
KV = re.compile(r"(\w+)=(-?\d+)")


def inventory(path):
    text = path.read_text()
    tables = {key: {} for key in TABLES}
    section = None
    comps = {}
    current = None
    for line in text.splitlines():
        if line in tables:
            section = line
            continue
        if section and line[:1].isdigit():
            key, value = line.split(" ", 1)
            tables[section][int(key)] = value
            continue
        if section:
            section = None
        m = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if m:
            current = m.group(2)
            comps[current] = []
        elif line == "}":
            current = None
        elif current and line.lstrip().startswith(("%", "ROOT %")) and " = " in line:
            comps[current].append(line)

    def location(fid):
        frame = dict((k, int(v)) for k, v in KV.findall(tables["StackFrames"].get(fid, "")))
        loc = dict((k, int(v)) for k, v in KV.findall(tables["FileLocations"].get(frame.get("file_location_id"), "")))
        file = tables["FileNames"].get(loc.get("file_name_id"), "UNKNOWN").strip('"')
        fn = tables["FunctionNames"].get(loc.get("function_name_id"), "UNKNOWN").strip('"')
        return file + ":" + str(loc.get("line")) + ":" + fn

    def sources(name):
        counts = Counter()
        for line in comps.get(name, []):
            for frame in re.findall(r"stack_frame_id=(\d+)", line):
                counts[location(int(frame))] += 1
        return counts

    fusions = []
    copies = Counter()
    for comp, lines in comps.items():
        for line in lines:
            match = re.search(r"\sfusion\(.*\bcalls=%?([\w.\-]+)", line)
            if match:
                name = re.search(r"%([\w.\-]+) = ", line).group(1)
                counts = sources(match.group(1))
                fusions.append({"name": name, "caller": comp, "body": match.group(1),
                                "sources": dict(counts), "body_instructions": len(comps[match.group(1)]),
                                "source_instruction_count": sum(counts.values())})
            if re.search(r"\scopy\(", line):
                frame = re.search(r"stack_frame_id=(\d+)", line)
                copies[location(int(frame.group(1))) if frame else "UNKNOWN"] += 1
    by_file = Counter()
    for row in fusions:
        source = Counter()
        for loc, n in row["sources"].items():
            source[loc.split(":", 1)[0]] += n
        row["dominant_source_file"] = source.most_common(1)[0][0] if source else "UNKNOWN"
        by_file[row["dominant_source_file"]] += 1
    assert fusions and any(row["sources"] for row in fusions), "missing source attribution"
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "computations": len(comps), "static_fusion_calls": len(fusions),
            "fusion_calls_by_dominant_source_file": dict(by_file.most_common()),
            "static_copy_instructions_by_source": dict(copies.most_common()), "fusions": fusions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    old, new = inventory(a.old), inventory(a.new)
    files = old["fusion_calls_by_dominant_source_file"].keys() | new["fusion_calls_by_dominant_source_file"].keys()
    changes = [{"source_file": file, "old": old["fusion_calls_by_dominant_source_file"].get(file, 0),
                "new": new["fusion_calls_by_dominant_source_file"].get(file, 0)} for file in files]
    changes.sort(key=lambda row: -(row["new"] - row["old"]))
    result = {"schema": "w44-perf.static_hlo.v1", "scope": __doc__, "old": old, "new": new,
              "changes": changes}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print("static fusion calls", old["static_fusion_calls"], new["static_fusion_calls"])
    for row in changes[:12]:
        print(row)


if __name__ == "__main__":
    main()
