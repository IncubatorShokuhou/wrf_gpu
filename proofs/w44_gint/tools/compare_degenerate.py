"""Compare the frozen-integrity degenerate frames of two candidates against the SAME reference (e.g. GPU vs CPU-twin), CPU only.
usage: python3 compare_degenerate.py GPU_INTEG.json TWIN_INTEG.json DOMAINS(d01,..) [MAX_FRAME_LABEL e.g. 2026-05-05_18]
Prints shared / GPU-only / twin-only (domain, field, frame) triples; frames after MAX_FRAME_LABEL are ignored on both sides
(a truncated twin run)."""
import json, sys

g, t, doms = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3].split(",")
lim = sys.argv[4] if len(sys.argv) > 4 else None


def frames(j):
    out = set()
    for d in doms:
        v = j["domains"].get(d, {})
        for f, hs in v.get("degenerate_fields", {}).items():
            for h in hs:
                lab = h["file"][11:24]
                if lim is None or lab <= lim:
                    out.add((d, f, lab))
    return out


G, T = frames(g), frames(t)
res = {"domains": doms, "max_frame": lim, "gpu": len(G), "twin": len(T), "shared": sorted(G & T), "gpu_only": sorted(G - T),
       "twin_only": sorted(T - G)}
print(json.dumps(res, indent=1))
