"""S01 parse: pristine RRTMG_SWRAD outputs on the Swiss columns -> one registered fixture (operands + radii + oracle).

usage: python s01_parse.py S01_DIR FIXTURE_DIR
The fixture holds RE01-style operands (in_*, col_re_cloud/ice/snow [m], category, ji, stamp, cpu_swdown) and AE01-style
per-arm oracle outputs (arm_julian/aer_opt/scale; solcon, declin, coszen, julian, o3, aerod, hsw ... swddir, swddif)
so tests/v025/w41_swddir/swddir_oracle_runner.py --swiss can feed sw_oracle_runner._solve/_metrics unchanged.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ae01_parse", ROOT / "proofs/w41_aerosol/ae01_parse.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def parse_driver(path: Path) -> dict:
    """ae01_parse.parse_driver's record layout (same driver output) on an explicit file."""
    r = P.Reader(path)
    ncol, nz, narm = r.take("<i4", 3)
    keys = ["solcon", "declin", "coszen", "julian", "o3", "aerod", "hsw", "hswc", "swup", "swupc", "swdn", "swdnc",
            *P.SCALARS]
    out = {k: [] for k in keys}
    for _ in range(ncol):
        for _ in range(narm):
            for k, v in zip(("solcon", "declin", "coszen", "julian"), r.take("<f4", 4)):
                out[k].append(v)
            out["o3"].append(r.take("<f4", nz))
            out["aerod"].append(r.take("<f4", nz * 6, (nz, 6), "F"))
            out["hsw"].append(r.take("<f4", nz)); out["hswc"].append(r.take("<f4", nz))
            for k in ("swup", "swupc", "swdn", "swdnc"):
                out[k].append(r.take("<f4", nz + 2))
            for k, v in zip(P.SCALARS, r.take("<f4", len(P.SCALARS))):
                out[k].append(v)
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {k: np.asarray(v).reshape((ncol, narm) + np.asarray(v[0]).shape) for k, v in out.items()}


def main(d: Path, fixtures: Path) -> None:
    ops = dict(np.load(d / "s01_operands.npz"))
    arms = json.loads((d / "s01_arms.json").read_text())["arms"]
    sw = parse_driver(d / "s01_output.bin")
    ncol, nz = sw["hsw"].shape[0], sw["hsw"].shape[2]
    narm = sw["hsw"].shape[1]
    radii = np.fromfile(d / "s01_output.bin.radii", "<f4").reshape(ncol, 3, nz)
    assert ncol == ops["in_t"].shape[0] and nz == ops["in_t"].shape[1] and narm == len(arms)
    day = sw["coszen"][:, 0] > 0
    checks = {
        "ncol": int(ncol), "nz": int(nz), "narm": int(narm),
        "categories": {c: int((ops["category"] == c).sum()) for c in np.unique(ops["category"])},
        "own_call_day_columns": int(day.sum()),
        "radii_um_ranges": {k: [float(radii[:, i].min() * 1e6), float(radii[:, i].max() * 1e6)]
                            for i, k in enumerate(("cloud", "ice", "snow"))},
        # WRF history SWDOWN at the frame holds the last radiation call (radt 10 min earlier): statistical sanity only
        "cpu_history_swdown_vs_oracle_corr": float(np.corrcoef(ops["cpu_swdown"][day], sw["swdnb"][day, 0])[0, 1]),
        "own_call_day_swddir_max": float(sw["swddir"][day, 0].max()),
    }
    fixtures.mkdir(parents=True, exist_ok=True)
    path = fixtures / "swddir_swiss_s01_v1.npz"
    np.savez_compressed(path, arm_julian=np.asarray([a[0] for a in arms], np.float32),
                        arm_aer_opt=np.asarray([a[1] for a in arms], np.int32),
                        arm_scale=np.asarray([a[2] for a in arms], np.float32),
                        col_re_cloud=radii[:, 0], col_re_ice=radii[:, 1], col_re_snow=radii[:, 2], **ops, **sw)
    summary = {"checks": checks, "arms": arms, "fixture": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
               "raw": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (d / "s01_input.bin", d / "s01_output.bin", d / "s01_output.bin.radii")}}
    Path(__file__).with_name("s01_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
