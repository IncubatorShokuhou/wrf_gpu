"""AE07 parse: pristine aer_opt=3 oracle outputs -> registered test fixture (+ self-checks vs AE01/RE01 arm A).

usage: python ae07_parse.py AE01_DIR FIXTURE_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ae01_parse import RE01, SCALARS, Reader  # noqa: E402


def parse(d: Path) -> dict:
    r = Reader(d / "ae07_output.bin")
    ncol, nz, narm = r.take("<i4", 3)
    keys = ["solcon", "declin", "coszen", "julian", "taod5503d", "taod5502d", "tauaer", "ssaaer", "asyaer",
            "hsw", "hswc", "swup", "swupc", "swdn", "swdnc", *SCALARS]
    out = {k: [] for k in keys}
    for _ in range(ncol):
        for _ in range(narm):
            for k, v in zip(("solcon", "declin", "coszen", "julian"), r.take("<f4", 4)):
                out[k].append(v)
            out["taod5503d"].append(r.take("<f4", nz))
            out["taod5502d"].append(r.take("<f4", 1)[0])
            for k in ("tauaer", "ssaaer", "asyaer"):
                out[k].append(r.take("<f4", nz * 14, (nz, 14), "F"))
            out["hsw"].append(r.take("<f4", nz)); out["hswc"].append(r.take("<f4", nz))
            for k in ("swup", "swupc", "swdn", "swdnc"):
                out[k].append(r.take("<f4", nz + 2))
            for k, v in zip(SCALARS, r.take("<f4", len(SCALARS))):
                out[k].append(v)
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {k: np.asarray(v).reshape((ncol, narm) + np.asarray(v[0]).shape) for k, v in out.items()}


def main(d: Path, fixtures: Path) -> None:
    sw = parse(d)
    arms = json.loads((d / "ae07_arms.json").read_text())["arms"]
    aer = dict(np.load(d / "ae07_aerosol.npz"))
    re01 = np.load(RE01)
    checks = {}
    for k in ("hsw", "hswc", "swup", "swupc", "swdn", "swdnc", "gsw", "swupt", "swdnb", "swddir", "swddif", "swdownc"):
        checks[f"arm0_{k}_bitwise_RE01_A"] = bool(np.array_equal(sw[k][:, 0].view(np.uint32), re01[f"A_{k}"].view(np.uint32)))
    day = sw["coszen"][:, 1] > 0.05
    clear = re01["category"] == "clear"
    checks["taod5502d_median_per_arm"] = [float(np.median(sw["taod5502d"][:, a])) for a in range(len(arms))]
    checks["layer_floor_1e-3_fraction_arm1"] = float((sw["taod5503d"][:, 1] == np.float32(1e-3)).mean())
    checks["clear_day_swdnb_reduction_arm1_wm2"] = [float(x) for x in np.percentile(
        (sw["swdnb"][:, 0] - sw["swdnb"][:, 1])[day & clear], [0, 50, 100])]
    checks["clear_day_swdnb_reduction_arm2_x10_wm2"] = [float(x) for x in np.percentile(
        (sw["swdnb"][:, 0] - sw["swdnb"][:, 2])[day & clear], [0, 50, 100])]
    fixtures.mkdir(parents=True, exist_ok=True)
    path = fixtures / "thompson_aer_sw_0227_v1.npz"
    np.savez_compressed(path, arm_julian=np.asarray([a[0] for a in arms], np.float32),
                        arm_aer_opt=np.asarray([a[1] for a in arms], np.int32),
                        arm_scale=np.asarray([a[2] for a in arms], np.float32),
                        in_qnwfa=aer["qnwfa"], in_qnifa=aer["qnifa"], **sw)
    summary = {"checks": checks, "arms": arms,
               "fixture": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
               "raw": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(d.glob("ae07_*.bin"))}}
    (Path(__file__).with_name("ae07_summary.json")).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
