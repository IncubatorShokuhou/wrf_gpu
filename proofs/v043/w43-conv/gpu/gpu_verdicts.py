"""CG verdicts: per-arm NUMERIC verdicts of the GPU run (launcher rc / process rc are NOT scientific passes; critic
cr-w43-conv). Every arm must show its EXPLICIT expected coverage, else FAIL (critic delta: all() over an empty result is
not a pass): expected frames = the 11 frames under analysis/data/frames, expected fields/species/columns/stages per arm
below; every expected entry must exist and pass. CG01 is re-scored with the E200-fixed frozen CT03 scorer (CPU).
usage: gpu_verdicts.py <outd>"""
import glob, json, os, subprocess, sys
from pathlib import Path
O = Path(sys.argv[1]); W = Path(__file__).resolve().parents[2]
FRAMES = sorted(os.path.basename(p) for p in glob.glob(str(W / "analysis/data/frames/*.npz")))
assert len(FRAMES) == 11, FRAMES
N_COLS = 586
SIXTH = [f"{f}_{m}" for f in ("U", "V", "W", "T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN") for m in ("mono", "nomono")]
MYNN = ("rthblten", "rqvblten", "rublten", "rvblten", "rqcblten", "rqiblten")
PD = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE")
SL = ("ust", "hfx", "lh", "qsfc", "br", "zol", "mol", "psim", "psih", "rmol", "u10", "v10", "t2", "th2", "regime", "znt")
KH = ("d11", "d22", "d12", "xkmh", "xkhh")
HD = ("U", "V", "W", "T", "QVAPOR", "QRAIN", "QGRAUP", "QNRAIN")
AC_FIELDS = 14
v, cov = {}, {}


def load(name):
    return json.load(open(O / name))


def per_frame(name, check):
    d = load(name)
    missing = [f for f in FRAMES if f not in d]
    ok_frames = [f for f in FRAMES if f in d and check(d[f])]
    cov[name] = {"expected_frames": len(FRAMES), "present": len(FRAMES) - len(missing), "passing": len(ok_frames), "missing": missing}
    return not missing and len(ok_frames) == len(FRAMES)


def arm(name, fn):
    try:
        v[name] = bool(fn())
    except Exception as e:  # noqa: BLE001
        v[name] = False; v[name + "_error"] = repr(e)[:300]


def cg01():
    rc = subprocess.run([sys.executable, str(W / "analysis/thompson/score.py"), str(W / "analysis/data/conv_columns.npz"),
                         str(O / "cg01_thompson_gpu.npz"), str(O / "cg01_score.json")], capture_output=True, text=True,
                        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"}).returncode
    s = load("cg01_score.json")
    cov["cg01"] = {"expected_columns": N_COLS, "scored_columns": s.get("census", {}).get("n")}
    return rc == 0 and s["valid"] and s["passed"] and s["n_fail"] == 0 and s["census"]["n"] == N_COLS


arm("CG01_thompson", cg01)
arm("CG02_sixth", lambda: per_frame("cg02_sixth_gpu.json", lambda r: all(k in r and r[k]["abs"]["passed"] and r[k]["rel"]["passed"] for k in SIXTH)))
arm("CG03_mynn", lambda: per_frame("cg03_mynn_gpu.json", lambda r: r.get("n", 0) > 0 and all(k in r and r[k]["pass_frozen_1e-3"] for k in MYNN))
    and load("cg03_mynn_gpu.json")["ALL_PASS_FROZEN"] is True)
arm("CG04_pd", lambda: per_frame("cg04_pd_gpu.json", lambda r: all(k in r and r[k]["passed"] for k in PD))
    and all(load("cg04_pd_gpu.json")["COVERAGE_species_present_and_limited_frames"][k] >= 1 for k in PD))  # every species covered somewhere
arm("CG05_mynnsl", lambda: per_frame("cg05_mynnsl_gpu.json", lambda r: r.get("n", 0) > 0 and all(k in r and r[k]["n_fail"] == 0 for k in SL))
    and load("cg05_mynnsl_gpu.json")["ALL_PASS_FROZEN"] is True)
arm("CG06_kh", lambda: per_frame("cg06_kh_gpu.json", lambda r: all(k in r and r[k]["passed"] for k in KH)))
arm("CG07_hdiff", lambda: per_frame("cg07_hdiff_gpu.json", lambda r: all(k in r and r[k]["abs"]["passed"] and r[k]["rel"]["passed"] for k in HD)))


def cg08():
    d = load("cg08_acoustic_gpu.json")
    st = d["stages"]
    shape = [len(s["steps"]) for s in st]
    nf = [len(e) for s in st for e in s["steps"].values()]
    cov["cg08"] = {"expected_stage_steps": [1, 2, 4], "stage_steps": shape, "fields_per_step": sorted(set(nf)),
                   "coefficients": [len(s["coefficients"]) for s in st]}
    return (shape == [1, 2, 4] and all(n == AC_FIELDS for n in nf) and all(len(s["coefficients"]) > 0 for s in st)
            and all(s["passed"] for s in st) and d["passed"] is True)


arm("CG08_acoustic", cg08)
v["ALL_PASS"] = all(v[k] for k in list(v) if k.startswith("CG") and not k.endswith("_error"))
(O / "VERDICTS.json").write_text(json.dumps({"verdicts": v, "coverage": cov}, indent=1))
print(" ".join(f"{k}={'PASS' if x else 'FAIL'}" for k, x in v.items() if not k.endswith("_error")))
