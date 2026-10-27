"""Pick the largest TTIR per (platform, module, kernel, site) from a merged dir into a rep dir."""
import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
best = {}
for l in open(f"{src}/index.jsonl"):
    m = json.loads(l); key = (m["platform"], m["kernel"], m["site"])
    if key not in best or m["ttir_chars"] > best[key]["ttir_chars"]: best[key] = m
with open(f"{dst}/index.jsonl", "w") as f:
    for m in best.values():
        fn = f"{m['platform']}_{m['ttir_sha16']}.ttir"; shutil.copy(f"{src}/{fn}", f"{dst}/{fn}"); f.write(json.dumps(m) + "\n")
print(len(best), "reps")
