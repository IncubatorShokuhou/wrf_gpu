"""Merge W5P01 capture dirs into one dir with a deduped index (platform, ttir sha)."""
import glob, json, os, shutil, sys
root = sys.argv[1]; capdir = sys.argv[2] if len(sys.argv) > 2 else "capture_nodes"; outname = sys.argv[3] if len(sys.argv) > 3 else "all"
out = os.path.join(root, outname); os.makedirs(out, exist_ok=True)
dirs = ([os.path.join(root, "smoke")] if outname == "all" else []) + sorted(glob.glob(os.path.join(root, capdir, "*/")))
seen = {}; fails = []; rows = []
for d in dirs:
    idx = os.path.join(d, "index.jsonl")
    if not os.path.exists(idx): continue
    for line in open(idx):
        m = json.loads(line)
        if m.get("status") == "LOWERED":
            fn = f"{m['platform']}_{m['ttir_sha16']}.ttir"
            if fn not in seen:
                shutil.copy(os.path.join(d, fn), os.path.join(out, fn)); seen[fn] = m; rows.append(m)
            else:
                seen[fn].setdefault("also_sites", []).append(m["site"]) if m["site"] != seen[fn]["site"] else None
        elif m.get("status") in ("LOWER_FAIL", "CAPTURE_FAIL"):
            fails.append(m)
with open(os.path.join(out, "index.jsonl"), "w") as f:
    for m in rows: f.write(json.dumps(m) + "\n")
json.dump(fails, open(os.path.join(out, "failures.json"), "w"), indent=1)
mods = {}
for m in rows:
    mods.setdefault(m["site"].split(":")[0], set()).add(m["kernel"])
print(len(rows), "unique TTIR", len(fails), "failures")
for k in sorted(mods): print(k, sorted(mods[k]))
