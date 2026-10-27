"""E100 floor: pristine WRF Thompson (default constants) under 1-REAL-ulp input perturbations.

Producer reproduction of the critic-supplied pristine ensemble protocol. Reuses the lane's default oracle build (build_oracle.py
--values '{}', which asserts byte-exact reproduction of the frozen fixture). Each member nudges
every NONZERO prognostic input (qv,qc,qr,qi,qs,qg,ni,nr,th) by +-1 float32 ulp (random sign,
fixed seed); zeros stay zero, p/pii/w/dz8w unchanged. Output: per-field spread vs the unperturbed run.
usage: python ens.py BUILD_DIR FIXTURE SELECTION OUT_JSON [members]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

build, fixture, selection, out = map(Path, sys.argv[1:5])
members = int(sys.argv[5]) if len(sys.argv) > 5 else 25
assert members >= 25
port_dir = Path(sys.argv[6]) if len(sys.argv) > 6 else None
z = dict(np.load(fixture, allow_pickle=False))
names = json.loads(selection.read_text())["selected_columns"]
idx = np.array([z["names"].tolist().index(n) for n in names])
ncol, nlev = len(idx), z["in_qv"].shape[1]
INPUTS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
PERTURB = INPUTS[:9]  # fixed field order; RNG does not depend on PYTHONHASHSEED
OUT = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")
SFC = ("rainncv", "snowncv", "graupelncv", "sr")


def write(path, fields):
    with path.open("wb") as stream:
        stream.write(np.array([ncol, nlev], ">i4").tobytes())
        stream.write(np.array([18.], ">f4").tobytes())
        for name in INPUTS:
            a = np.asarray(fields[name].T, ">f4")
            assert np.isfinite(a).all(), name
            stream.write(a.tobytes())


def run(tag, fields):
    write(build / f"in_{tag}.bin", fields)
    subprocess.run([str(build / "oracle"), f"in_{tag}.bin", f"out_{tag}.bin"], cwd=build, check=True,
                   stdout=subprocess.DEVNULL, timeout=30)
    raw = np.fromfile(build / f"out_{tag}.bin", ">f4")
    assert np.isfinite(raw).all(), tag
    res, off = {}, 0
    for name in OUT:
        res[name] = raw[off:off + ncol * nlev].reshape(nlev, ncol).T.astype(np.float32)
        off += ncol * nlev
    for name in SFC:
        res[name] = raw[off:off + ncol].astype(np.float32)
        off += ncol
    assert off == raw.size
    (build / f"in_{tag}.bin").unlink()
    (build / f"out_{tag}.bin").unlink()
    return res


def ulps(a, b):
    def ordered(x):
        bits = x.view(np.uint32).astype(np.int64)
        return np.where(bits & 0x80000000, -(bits & 0x7FFFFFFF), bits)  # monotone, +0 == -0
    return np.abs(ordered(a) - ordered(b))


base_in = {name: np.asarray(z["in_" + name][idx], np.float32) for name in INPUTS}
base = run("base", base_in)
ref_ok = {name: base[name].tobytes() == np.asarray(z["wrf_" + name][idx], np.float32).tobytes()
          for name in OUT if "wrf_" + name in z}
assert all(ref_ok.values()), ref_ok
envelope = {name: np.zeros_like(base[name], dtype=np.float64) for name in OUT + SFC}
rng = np.random.default_rng(20261024)
stats = {name: dict(max_ulp=[], max_abs=[], changed_cells=[]) for name in OUT + SFC}
for m in range(members):
    pert = dict(base_in)
    for name in PERTURB:
        a = base_in[name]
        sign = rng.choice(np.array([-np.inf, np.inf], np.float32), size=a.shape)
        pert[name] = np.where(a != 0, np.nextafter(a, sign), a).astype(np.float32)
    res = run(f"m{m:02d}", pert)
    for name in OUT + SFC:
        error = np.abs(res[name].astype(np.float64) - base[name])
        envelope[name] = np.maximum(envelope[name], error)
        u = ulps(res[name], base[name])
        stats[name]["max_ulp"].append(int(u.max()))
        stats[name]["max_abs"].append(float(np.max(np.abs(res[name].astype(np.float64) - base[name]))))
        stats[name]["changed_cells"].append(int(np.count_nonzero(u)))
summary = {name: dict(members=members, min_member_max_ulp=min(v["max_ulp"]), median_member_max_ulp=float(np.median(v["max_ulp"])),
                      max_member_max_ulp=max(v["max_ulp"]), median_member_max_abs=float(np.median(v["max_abs"])),
                      max_member_max_abs=max(v["max_abs"]), median_changed_cells=float(np.median(v["changed_cells"])))
           for name, v in stats.items()}
result = dict(platform="cpu", gpu_executed=False, oracle="pristine WRF module_mp_thompson.f90 default constants, lane build_oracle.py",
              columns=ncol, levels=nlev, dt=18., members=members, seed=20261024,
              perturbation="+-1 float32 ulp on every nonzero qv,qc,qr,qi,qs,qg,ni,nr,th input cell",
              base_reproduces_frozen_fixture=ref_ok, summary=summary, per_member=stats)
# E100's absolute floor is tied to the field maximum, as in the frozen
# increment rule. Register the max over the fixed 25-member ensemble before
# viewing the candidate. Also disclose the stricter cellwise envelope counts.
result["floor_definition"] = "maximum absolute pristine output change per field over the fixed input ensemble; cellwise exceedances disclosed separately"
if port_dir is not None:
    def physical(label):
        raw=dict(np.load(port_dir/(label+".npz"),allow_pickle=False))
        assert all(np.isfinite(a).all() for a in raw.values())
        fields={name:raw[str(i)].astype(np.float64) for i,name in enumerate(OUT[:8])}
        fields["th"]=(raw["10"].astype(np.float32)/z["in_pii"][idx].astype(np.float32)).astype(np.float64)
        fields["rainncv"]=sum(raw[str(i)].astype(np.float64) for i in (16,17,18,19))
        fields["snowncv"]=raw["17"].astype(np.float64)+raw["19"].astype(np.float64)
        fields["graupelncv"]=raw["16"].astype(np.float64)
        return fields
    off,on=physical("off"),physical("dynamic")
    comparison={}
    for name in off:
        error=np.abs(on[name]-off[name])
        floor=float(envelope[name].max())
        comparison[name]={"default_max_abs":float(error.max()),"ensemble_field_floor":floor,
                          "within_field_floor":bool(error.max()<=floor),
                          "cells_above_cellwise_envelope":int(np.count_nonzero(error>envelope[name])),
                          "changed_cells":int(np.count_nonzero(error)),
                          "wrf_field_max_abs":float(np.abs(base[name]).max())}
    result["combined_default_vs_floor"]=comparison
    result["within_all_oracle_field_floors"]=all(row["within_field_floor"] for row in comparison.values())
    result["scope_limit"]="Original oracle returns 9 prognostics and aggregate surface precipitation. Internal Ns/Ng and individual cloud/ice/rain/snow precipitation classes are reported by combined_defaults, not claimed as directly observed pristine ensemble outputs."
np.savez_compressed(out.with_suffix(".npz"),**envelope)

out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(ref_ok=ref_ok, summary=summary), indent=1))

if port_dir is not None:
    assert result["within_all_oracle_field_floors"], result["combined_default_vs_floor"]
