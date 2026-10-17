"""Pristine-WRF CANWATER oracle for GPUWRF_NOAH_CANWATER_ELAI (V033-W2, b-thompson/b-noahmp). CPU only, no JAX.

Slices PHENOLOGY (:1255-1358) and CANWATER (:6265-6394) VERBATIM from the pristine module_sf_noahmplsm.F and checks
their SHA256 values. Builds canwater_oracle.F90 (build.sh, compiled pristine objects + real MPTABLE) in a lane dir
and runs a column set designed to exercise snow burial of the canopy. It covers real MODIS categories (tall
ENF/DBF/mixed, short-canopy SNOWHC branch shrub/grass/crop/tundra, and urban/barren with no veg), SNOWH from
0 to 12 m (unburied, partial burial, full burial so ELAI+ESAI = 0), and canopy water as liquid, ice, mixed
and above capacity. TV below, at and above TFRZ drives the freeze/melt TV pin; FCEV covers dew/frost,
zero and evaporation/sublimation. It writes the inputs plus WRF's
LAI/SAI/ELAI/ESAI/FB/FVEG/BDFALL/CANLIQ/CANICE/TV/FWET/CMC/ECAN/ETRAN/QFRZC/QMELTC to the test fixture.
usage: make_oracle.py <lane_build_dir> <fixture.npz>"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F")
SLICES = {"PHENOLOGY": (1255, 1358), "CANWATER": (6265, 6394)}
HERE = Path(__file__).resolve().parent
B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TFRZ = np.float32(273.16)

lines = PRISTINE.read_text().splitlines(keepends=True)
inc, prov = [], {"pristine": str(PRISTINE), "pristine_sha256": hashlib.sha256(PRISTINE.read_bytes()).hexdigest()}
for name, (a, b) in SLICES.items():
    body = "".join(lines[a - 1:b])
    assert body.lstrip().startswith(f"SUBROUTINE {name}") and lines[b - 1].strip() == f"END SUBROUTINE {name}", name
    prov[name] = {"lines": [a, b], "sha256": hashlib.sha256(body.encode()).hexdigest()}
    inc.append(body)
B.mkdir(parents=True, exist_ok=True)
(B / "cwe_routines.inc.src").write_text("".join(inc))
subprocess.run(["bash", str(HERE / "build.sh"), str(B), str(B / "cwe_routines.inc.src")], check=True)

VEG = [1, 4, 5, 7, 8, 10, 12, 13, 16, 18, 19]
SNOWH = [0.0, 0.02, 0.1, 0.4, 1.0, 3.0, 6.0, 12.0]   # <= 12 m: short-canopy SNOWHC = HVT*exp(-SNOWH/0.2) stays a normal REAL
CAN = [(0.0, 0.0), (0.03, 0.0), (0.0, 0.04), (0.3, 0.1), (0.1, 0.3), (2.0, 0.0), (0.0, 2.0)]
TV = [270.5, 273.16, 275.5]
FCEV = [-8.0, 0.0, 15.0]
SHD = [0.02, 0.45, 0.9]
rows = []
for iv, v in enumerate(VEG):
    for s in SNOWH:
        for cl, ci in CAN:
            for t in TV:
                for f in FCEV:
                    rows.append((v, s, cl, ci, t, f, SHD[iv % 3]))
n = len(rows)
col = {k: np.array([r[i] for r in rows]) for i, k in enumerate(("vegtyp", "snowh", "canliq", "canice", "tv", "fcev", "shdmax"))}
f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
inp = dict(vegtyp=col["vegtyp"].astype(np.int32), lat=f32(np.full(n, np.radians(28.0))), julian=f32(np.full(n, 45.5)),
           snowh=f32(col["snowh"]), tv0=f32(col["tv"]), troot=f32(col["tv"]), shdmax=f32(col["shdmax"]),
           sfctmp=f32(col["tv"] - 1.0), canliq=f32(col["canliq"]), canice=f32(col["canice"]), tv=f32(col["tv"]),
           fcev=f32(col["fcev"]), fctr=f32(np.where(col["fcev"] >= 0, 20.0, 0.0)), tg=f32(col["tv"]))
inp["frz"] = (inp["tv0"] <= TFRZ).astype(np.int32)       # ENERGY :2210-2216 (TV entering ENERGY <= TFRZ -> frozen)
DT, YEARLEN = np.float32(18.0), 365
rec = np.dtype([("vegtyp", ">i4"), ("lat", ">f4"), ("julian", ">f4"), ("snowh", ">f4"), ("tv0", ">f4"), ("troot", ">f4"),
                ("shdmax", ">f4"), ("sfctmp", ">f4"), ("canliq", ">f4"), ("canice", ">f4"), ("tv", ">f4"), ("fcev", ">f4"),
                ("fctr", ">f4"), ("tg", ">f4"), ("frz", ">i4")])
arr = np.zeros(n, rec)
for k in rec.names:
    arr[k] = inp[k]
with open(B / "cwe_in.bin", "wb") as fh:
    fh.write(np.array([n], ">i4").tobytes() + np.array([DT], ">f4").tobytes() + np.array([YEARLEN], ">i4").tobytes())
    fh.write(arr.tobytes())
subprocess.run(["taskset", "-c", "12", "./canwater_oracle.exe"], cwd=B, check=True)
OUTN = ("lai", "sai", "elai", "esai", "fb", "fveg", "bdfall", "canliq", "canice", "tv", "fwet", "cmc", "ecan", "etran",
        "qfrzc", "qmeltc")
raw = np.fromfile(B / "cwe_out.bin", ">f4")
assert raw.size == n * len(OUTN), (raw.size, n)
raw = raw.reshape(n, len(OUTN)).astype(np.float32)
assert np.isfinite(raw).all()
wrf = {f"wrf_{k}": raw[:, i] for i, k in enumerate(OUTN)}
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **{f"in_{k}": v for k, v in inp.items()}, dt=DT, yearlen=np.int32(YEARLEN), **wrf,
                    provenance=np.array(json.dumps(prov)))
buried = (raw[:, 2] + raw[:, 3]) != (raw[:, 0] + raw[:, 1])
print(f"{n} columns -> {OUT}; ELAI+ESAI != LAI+SAI in {int(buried.sum())}; ELAI+ESAI == 0 < LAI+SAI in "
      f"{int(((raw[:, 2] + raw[:, 3]) == 0).sum() - ((raw[:, 0] + raw[:, 1]) == 0).sum())}; canopy water > 0 in "
      f"{int(((raw[:, 7] + raw[:, 8]) > 0).sum())}; prov {json.dumps(prov)[:300]}")
