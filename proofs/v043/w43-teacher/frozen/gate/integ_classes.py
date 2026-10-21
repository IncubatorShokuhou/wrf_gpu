"""Integrity verdict against the REGISTERED exception classes (manager 2026-10-06T01:42:32Z): {PASS | PASS-with-registered-exceptions
[class + frame list] | FAIL-new-class}. Only FAIL-new-class blocks. Inputs: the frozen wn3_integrity JSON (degenerate fields per frame:
GPU constant value + CPU range) and, for the R4 cell watch, r4_eval.py data (classify(integ, r4)). Class definitions (manager MAILBOX):
  R2 trace ice/snow     QICE, QSNOW degenerate with GPU 0 and CPU |max| <= 1e-9                         (2026-10-05T03:31:10Z)
  R4 canopy-water/evaporation timing  ECAN, EVC, CANLIQ, CANWAT, FWET                                  (03:31:10Z, 2026-10-06T01:42:32Z)
     CELL WATCH (2026-10-06T05:29:12Z + 05:31:13Z, = integrate RC33P/r4_watch.py; replaces the retired 1 W/m2 EVC watch): in the
     flagged cells (CPU != 0 in any R4 field degenerate in that frame): CPU CANLIQ <= 1e-2 mm AND CPU CANWAT <= 1e-2 mm AND CPU AND GPU
     hourly rain (RAINNC+RAINC increment from the previous GPU frame) <= 1e-2 mm AND GPU canopy dried (GPU CANLIQ == CANWAT == 0).
     Data = r4_eval.py JSON captured before the GPU frames are deleted; no data for that frame -> FAIL-new-class
  R5 snow/canopy threshold-timing traces  CANICE, QSNOWXY: HARD cap CPU |max| <= 1e-6 (05:29:12Z/05:31:13Z; above = FAIL-new-class)
     + ACSNOM (04:40:07Z): FAIL-new-class if CPU ACSNOM > 0.1 kg/m2 while the GPU field is 0 domain-wide
COMPANIONS (05:16:51Z, window 05:18:26Z): valid only within +-1 h of a frame in the SAME domain flagged by an ACCEPTED parent field of
the same class, and only under the watch (CPU |max|): R2 + QNICE <= 1e4 /kg, QGRAUP <= 1e-9 kg/kg; R4 + QRAINXY <= 1e-6 mm;
R5 + SNOW <= 0.1 kg/m2, SNOWC <= 0.01, SNOWH <= 1e-3 m, TAUSS (any). Outside the window or above the watch = FAIL-new-class.
Anything else — another degenerate field, missing variables vs CPU, failing globals, pairing — is FAIL-new-class."""
import datetime
import glob
import os

import numpy as np

CLASSES = {"QICE": "R2", "QSNOW": "R2", "ECAN": "R4", "EVC": "R4", "CANLIQ": "R4", "CANWAT": "R4", "FWET": "R4",
           "CANICE": "R5", "QSNOWXY": "R5", "ACSNOM": "R5"}
COMPANIONS = {"QNICE": ("R2", 1e4), "QGRAUP": ("R2", 1e-9), "QRAINXY": ("R4", 1e-6),
              "SNOW": ("R5", 0.1), "SNOWC": ("R5", 0.01), "SNOWH": ("R5", 1e-3), "TAUSS": ("R5", float("inf"))}
R4_CAN_MAX = 1e-2   # mm, CPU CANLIQ and CANWAT in the flagged cells
R5_CAP = 1e-6       # CANICE / QSNOWXY CPU |max|


def frame(f):
    return f["file"][11:24]  # 2026-06-03_12


def ftime(label):
    return datetime.datetime.strptime(label, "%Y-%m-%d_%H")


def near_parent(parents, cls, label):
    t = ftime(label)
    return any(c == cls and abs((ftime(pl) - t).total_seconds()) <= 3600 for c, pl in parents)


R4_RAIN_MAX = 1e-2  # mm per hour, CPU and GPU, flagged cells


def r4_watch(r4, dom, label, fields):
    """(ok, detail) of the R4 cell watch for one flagged frame from r4_eval.py data; fields = R4 fields degenerate in that frame."""
    fr = (r4 or {}).get("frames", {}).get(f"{dom} {label}")
    if fr is None:
        return False, "not evaluated (no r4_eval data for this frame)"
    if fr.get("prev") is None:
        return False, "not evaluable (first frame, no previous hour)"
    rows = [fr["fields"][f] for f in fields if fr["fields"].get(f, {}).get("cells")]
    if not rows:
        return True, "0 flagged cells"
    m = {k: max(r[k] for r in rows) for k in ("cpu_canliq_max", "cpu_canwat_max", "cpu_rain_1h_max", "gpu_rain_1h_max", "gpu_canliq_max", "gpu_canwat_max")}
    ok = (m["cpu_canliq_max"] <= R4_CAN_MAX and m["cpu_canwat_max"] <= R4_CAN_MAX and m["cpu_rain_1h_max"] <= R4_RAIN_MAX
          and m["gpu_rain_1h_max"] <= R4_RAIN_MAX and m["gpu_canliq_max"] == 0 and m["gpu_canwat_max"] == 0)
    return ok, (f"<= {sum(r['cells'] for r in rows)} cells, CPU CANLIQ {m['cpu_canliq_max']:.3g} CANWAT {m['cpu_canwat_max']:.3g} mm, "
                f"rain 1 h CPU {m['cpu_rain_1h_max']:.3g} GPU {m['gpu_rain_1h_max']:.3g} mm, GPU canopy {m['gpu_canliq_max']:.3g}/{m['gpu_canwat_max']:.3g}")


def classify(integ, r4=None):
    out = {"verdict": "PASS", "domains": {}, "watch": [], "new": [], "r4_cells": []}
    for d, v in (integ.get("domains") or {}).items():
        dom, parents = {}, set()
        degen = v.get("degenerate_fields") or {}
        r4_by_frame = {}
        for fld in ("ECAN", "EVC", "CANLIQ", "CANWAT", "FWET"):
            for h in degen.get(fld) or []:
                r4_by_frame.setdefault(frame(h), []).append(fld)
        r4_done = {}
        # pass 1: parent fields
        for fld, hits in degen.items():
            if fld in COMPANIONS:
                continue
            for h in hits:
                cpu = max(abs(x) for x in h["cpu_range"])
                cls, why = CLASSES.get(fld), None
                if cls == "R2" and not (h.get("gpu_value") == 0.0 and cpu <= 1e-9):
                    cls, why = None, "R2 needs GPU 0 and CPU |max| <= 1e-9"
                elif fld == "ACSNOM" and h.get("gpu_value") == 0.0 and cpu > 0.1:
                    cls, why = None, "R5 ACSNOM guard: CPU > 0.1 kg/m2 while GPU 0"
                elif cls == "R5" and fld != "ACSNOM" and cpu > R5_CAP:
                    cls, why = None, f"R5 cap {R5_CAP:g}"
                elif cls == "R4":
                    if frame(h) not in r4_done:
                        r4_done[frame(h)] = r4_watch(r4, d, frame(h), r4_by_frame[frame(h)])
                        ok, detail = r4_done[frame(h)]
                        out["r4_cells"].append(f"{d} {'/'.join(r4_by_frame[frame(h)])}@{frame(h)}: {detail} -> {'R4' if ok else 'FAIL'}")
                    ok, detail = r4_done[frame(h)]
                    if not ok:
                        cls, why = None, f"R4 cell watch: {detail}"
                if cls is None:
                    out["new"].append(f"{d} {fld}@{frame(h)} (GPU {h.get('gpu_value')}, CPU |max| {cpu:.3g}" + (f"; {why}" if why else "") + ")")
                    continue
                parents.add((cls, frame(h)))
                dom.setdefault(cls, {}).setdefault(fld, []).append(frame(h))
        # pass 2: companions (need an ACCEPTED parent of the class within +-1 h, same domain)
        for fld, hits in degen.items():
            if fld not in COMPANIONS:
                continue
            pcls, bound = COMPANIONS[fld]
            for h in hits:
                cpu = max(abs(x) for x in h["cpu_range"])
                if near_parent(parents, pcls, frame(h)) and cpu <= bound:
                    dom.setdefault(pcls, {}).setdefault(fld + "(companion)", []).append(frame(h))
                else:
                    why = "no accepted parent-class flag within +-1 h" if not near_parent(parents, pcls, frame(h)) else f"CPU |max| {cpu:.3g} > watch {bound:g}"
                    out["new"].append(f"{d} {fld}@{frame(h)} ({pcls} companion invalid: {why}; GPU {h.get('gpu_value')})")
        if v.get("missing_variables_vs_cpu"):
            out["new"].append(f"{d} missing vs CPU {sorted(v['missing_variables_vs_cpu'])}")
        g = v.get("global_attrs") or {}
        if g.get("missing_required") or g.get("by_frame"):
            out["new"].append(f"{d} globals {g.get('missing_required') or sorted(g.get('by_frame'))[:3]}")
        if v.get("pairing_ok") is False:
            out["new"].append(f"{d} pairing failed")
        if dom:
            out["domains"][d] = dom
    if out["new"]:
        out["verdict"] = "FAIL-new-class"
    elif out["domains"]:
        out["verdict"] = "PASS-with-registered-exceptions"
    return out


def compact(c):
    ex = "; ".join(f"{d} {cls} " + ", ".join(f"{fld}@{'/'.join(fr)}" for fld, fr in flds.items())
                   for d, dom in c["domains"].items() for cls, flds in sorted(dom.items()))
    s = c["verdict"] + (f" [{ex}]" if ex else "")
    if c["new"]:
        s += " NEW: " + "; ".join(c["new"])
    if c.get("r4_cells"):
        s += " R4 cells: " + "; ".join(c["r4_cells"])
    if c["watch"]:
        s += " WATCH: " + "; ".join(c["watch"])
    return s
