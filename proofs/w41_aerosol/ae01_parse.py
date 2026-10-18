"""AE01 parse: pristine aer_opt=1 oracle outputs -> registered test fixtures (+ self-checks vs the RE01 fixture).

usage: python ae01_parse.py AE01_DIR FIXTURE_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RE01 = ROOT / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
SCALARS = ("gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc", "swdnb", "swdnbc", "swddir", "swddni",
           "swddif", "swdownc", "coszr")


class Reader:
    def __init__(self, path):
        self.buf, self.pos = Path(path).read_bytes(), 0

    def take(self, dtype, count, shape=None, order="C"):
        n = np.dtype(dtype).itemsize * count
        out = np.frombuffer(self.buf, dtype, count, self.pos)
        self.pos += n
        return out if shape is None else out.reshape(shape, order=order)


def parse_units(d: Path, units: dict) -> dict:
    lat, lon, jul, p = units["lat"], units["lon"], units["julian"], units["p"]
    r = Reader(d / "ae01_units_output.bin")
    out = {"rsrtaua": r.take("<f4", 84, (14, 6), "F"), "rsrpiza": r.take("<f4", 84, (14, 6), "F"),
           "rsrasya": r.take("<f4", 84, (14, 6), "F"), "pina": r.take("<f4", 12)}
    # Fortran aerodm(lon, level, lat, month, type) -> C [lat, lon, month, level, type]
    am = r.take("<f4", lon.size * 12 * lat.size * 12 * 6, (lon.size, 12, lat.size, 12, 6), "F")
    aerodm = np.ascontiguousarray(np.transpose(am, (2, 0, 3, 1, 4)))
    # the full (lat, lon, month, level, type) array is pinned by its SHA-256; a lat band is kept for diagnostics
    out["aerodm_sha256"] = np.asarray(hashlib.sha256(aerodm.tobytes()).hexdigest())
    out["aerodm_band"] = aerodm[lat.size // 2 - 2: lat.size // 2 + 3]
    out["time_unit"] = np.stack([r.take("<f4", 12) for _ in jul])
    ad = r.take("<f4", p.shape[0] * p.shape[1] * 6, (p.shape[0], p.shape[1], 6), "F")
    out["aerod"] = np.ascontiguousarray(ad)
    out["totaod"] = r.take("<f4", p.shape[0])
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {**{f"in_{k}": v for k, v in units.items()}, **out}


def parse_driver(d: Path) -> dict:
    r = Reader(d / "ae01_output.bin")
    ncol, nz, narm = r.take("<i4", 3)
    keys = ["solcon", "declin", "coszen", "julian", "o3", "aerod", "hsw", "hswc", "swup", "swupc", "swdn", "swdnc", *SCALARS]
    out = {k: [] for k in keys}
    for _ in range(ncol):
        for _ in range(narm):
            head = r.take("<f4", 4)
            for k, v in zip(("solcon", "declin", "coszen", "julian"), head):
                out[k].append(v)
            out["o3"].append(r.take("<f4", nz))
            out["aerod"].append(r.take("<f4", nz * 6, (nz, 6), "F"))
            out["hsw"].append(r.take("<f4", nz)); out["hswc"].append(r.take("<f4", nz))
            for k in ("swup", "swupc", "swdn", "swdnc"):
                out[k].append(r.take("<f4", nz + 2))
            for k, v in zip(SCALARS, r.take("<f4", len(SCALARS))):
                out[k].append(v)
    assert r.pos == len(r.buf), (r.pos, len(r.buf))
    return {k: np.asarray(v).reshape((ncol, narm) + np.asarray(v[0]).shape) for k, v in out.items()}


def main(d: Path, fixtures: Path) -> None:
    units = dict(np.load(d / "ae01_units_operands.npz"))
    u = parse_units(d, units)
    sw = parse_driver(d)
    arms = json.loads((d / "ae01_arms.json").read_text())["arms"]
    re01 = np.load(RE01)
    checks = {}
    # arm 0 (aer_opt 0, own julian) is RE01's arm A shortwave: bitwise
    for k in ("hsw", "hswc", "swup", "swupc", "swdn", "swdnc", "gsw", "swupt", "swdnb", "swddir", "swddif", "swdownc"):
        checks[f"arm0_{k}_bitwise_RE01_A"] = bool(np.array_equal(sw[k][:, 0].view(np.uint32), re01[f"A_{k}"].view(np.uint32)))
    checks["arm0_coszen_bitwise"] = bool(np.array_equal(sw["coszen"][:, 0], re01["col_coszen"]))
    checks["arm0_o3_bitwise"] = bool(np.array_equal(sw["o3"][:, 0], re01["col_o3"]))
    day = re01["col_coszen"] > 0
    checks["aerosol_changes_day_gsw"] = int((sw["gsw"][:, 1] != sw["gsw"][:, 0])[day].sum())
    checks["aerosol_night_gsw_zero"] = bool((sw["gsw"][~day] == 0).all())
    checks["clear_day_swdnb_reduction_wm2"] = [float(x) for x in np.percentile(
        (sw["swdnb"][:, 0] - sw["swdnb"][:, 1])[day & (re01["category"] == "clear")], [0, 50, 100])]
    checks["aod_total_per_arm_median"] = [float(np.median(sw["aerod"][:, a].sum(axis=(1, 2)))) for a in range(len(arms))]
    checks["dust_fraction_median"] = float(np.median(sw["aerod"][:, 1, :, 2].sum(1) / sw["aerod"][:, 1].sum(axis=(1, 2))))
    fixtures.mkdir(parents=True, exist_ok=True)
    sw_path, u_path = fixtures / "tegen_sw_0227_v1.npz", fixtures / "tegen_units_v1.npz"
    np.savez_compressed(sw_path, arm_julian=np.asarray([a[0] for a in arms], np.float32),
                        arm_aer_opt=np.asarray([a[1] for a in arms], np.int32),
                        arm_scale=np.asarray([a[2] for a in arms], np.float32), **sw)
    np.savez_compressed(u_path, **u)
    summary = {"checks": checks, "arms": arms,
               "fixtures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sw_path, u_path)},
               "raw": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("ae01_*.bin"))}}
    (Path(__file__).with_name("ae01_summary.json")).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
