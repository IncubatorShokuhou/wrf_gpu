"""List (default) or delete (--delete) ~/.triton/cache entries created by the W5P01 census:
created after 2026-10-17T11:20Z AND target backend hip OR cuda arch in {61, 75}. sm_120 entries are never touched."""
import json, os, shutil, sys, time, calendar
root = os.path.expanduser("~/.triton/cache"); t0 = calendar.timegm(time.strptime("2026-10-17T11:20:00", "%Y-%m-%dT%H:%M:%S"))
hits = []; total = 0
for d in os.listdir(root):
    p = os.path.join(root, d)
    if not os.path.isdir(p) or os.stat(p).st_mtime < t0: continue
    js = [f for f in os.listdir(p) if f.endswith(".json") and not f.startswith("__grp")]
    if not js: continue
    try: m = json.load(open(os.path.join(p, js[0])))
    except Exception: continue
    tg = m.get("target", {}); be, arch = tg.get("backend"), tg.get("arch")
    if be == "hip" or (be == "cuda" and arch in (61, 75)):
        sz = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)); hits.append((p, be, arch, m.get("name"), sz)); total += sz
print(len(hits), "entries", round(total / 1e6, 1), "MB")
if "--delete" in sys.argv:
    for p, *_ in hits: shutil.rmtree(p)
    print("deleted", len(hits))
