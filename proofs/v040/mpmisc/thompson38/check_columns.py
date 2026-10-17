"""Compare the composed mp38 candidate with six true original-WRF callers."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_hail.column import ThompsonHailColumnState, step_thompson_hail_column_with_precip


def parse(path):
    if not path.exists() and path.with_name("original_"+path.name).exists():
        path=path.with_name("original_"+path.name)
    if not path.exists() and (path.parent/"original_columns.npz").exists():
        prefix=path.stem+"__"
        with np.load(path.parent/"original_columns.npz",allow_pickle=False) as data:
            values={name.removeprefix(prefix):np.asarray(data[name]) for name in data.files if name.startswith(prefix)}
        columns={name:value.astype(np.float32).tolist() for name,value in values.items() if value.ndim==1}
        scalars={name:float(value) for name,value in values.items() if value.ndim==0}
        assert scalars["KX"]==40 and len(columns["QB_OUT"])==40
        return dict(columns=columns,scalars=scalars)
    columns, scalars = {}, {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$", line)
        if match:
            name, index, value = match.groups()
            columns.setdefault(name, {})[int(index)] = float(value)
        elif "=" in line:
            name, value = line.split("=", 1)
            if re.fullmatch(r"[A-Z0-9_]+", name):
                scalars[name] = float(value)
    assert scalars["KX"] == 40 and len(columns["QB_OUT"]) == 40
    return {"columns": {name: [values[k] for k in range(1, 41)] for name, values in columns.items()}, "scalars": scalars}


def surface_projection(precip, references):
    """Original public wrapper :1299-1308, including every REAL sum."""
    rain, snow, graupel, ice = (np.asarray(precip[name],np.float32) for name in ("rain","snow","graupel","ice"))
    total = ((rain+snow)+graupel)+ice
    solid = (snow+graupel)+ice
    values = dict(RAINNCV=total,SNOWNCV=snow+ice,GRAUPELNCV=graupel,SR=solid/(total+np.float32(1.e-12)))
    # All current single-step original callers start accumulators at zero.
    values.update(RAINNC=total,SNOWNC=snow+ice,GRAUPELNC=graupel)
    result={}
    for name,actual in values.items():
        expected=np.asarray([ref["scalars"][name] for ref in references],np.float32)
        difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        result[name]=dict(byte_equal=bool(np.array_equal(actual,expected)),finite=bool(np.isfinite(actual).all()),
                          matching=int(np.count_nonzero(actual==expected)),total=int(actual.size),
                          max_abs=float(difference.max()),actual=actual.tolist(),reference=expected.tolist())
    return result


def main(reference, out, cases=tuple(range(1,7))):
    refs = [parse(reference / f"case{c}.txt") for c in cases]
    dts = {r["scalars"]["DT"] for r in refs}
    assert len(dts) == 1, "batch only original columns with identical resolved DT"
    dt = dts.pop()
    cols = {name: np.asarray([r["columns"][name] for r in refs], np.float32) for name in refs[0]["columns"]}
    fields = {"qv": "QV", "qc": "QC", "qr": "QR", "qi": "QI", "qs": "QS", "qg": "QG", "Ni": "NI", "Nr": "NR", "Nc": "NC", "Ng": "NG", "qvolg": "QB", "nwfa": "NWFA", "nifa": "NIFA", "T": "T"}
    values = {field: jnp.asarray(cols[name + "_IN"]) for field, name in fields.items()}
    p, temp, qv = jnp.asarray(cols["P"]), values["T"], values["qv"]
    rho = np.float32(.622) * p / (np.float32(287.04) * temp * (qv + np.float32(.622)))
    w = jnp.asarray(cols["W_IN"]) if "W_IN" in cols else jnp.zeros_like(qv)
    state = ThompsonHailColumnState(**values, p=p, rho=rho, dz=jnp.asarray(cols["DELZ"]), w=w)
    start = time.monotonic()
    actual, precip = step_thompson_hail_column_with_precip(state, dt)
    actual = jax.tree.map(np.asarray, actual)
    precip = {name: np.asarray(a) for name, a in precip.items()}
    source_path=reference / "build.json"
    if not source_path.exists(): source_path=reference / "receipt.json"
    if not source_path.exists(): source_path=reference / ("original_build.json" if max(cases)<=6 else "fresh_original_build.json")
    result = {"platform": jax.devices()[0].platform, "cases":cases, "dt":dt, "wall_s": time.monotonic()-start, "fields": {}, "source": json.loads(source_path.read_text())}
    result["candidate_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[4],text=True).strip()
    result["prognostic_dtypes"] = {name:str(np.asarray(getattr(actual,name)).dtype) for name in actual.__slots__}
    result["precipitation_dtypes"] = {name:str(value.dtype) for name,value in precip.items()}
    assert all(value == "float32" for value in (*result["prognostic_dtypes"].values(),*result["precipitation_dtypes"].values()))
    for field, name in fields.items():
        a, b = np.asarray(getattr(actual, field)), cols[name + "_OUT"]
        if field == "T":
            # Original wrapper :1358 stores theta=T/pii; the oracle driver
            # dumps theta*pii. Preserve both REAL rounding points (E11).
            result["inner_temperature"] = a.tolist()
            a = (a / cols["PII"]) * cols["PII"]
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        spacing = np.abs(np.spacing(b)).astype(np.float64)
        result["fields"][field] = {"finite": bool(np.isfinite(a).all()), "byte_equal": bool(np.array_equal(a,b)), "max_abs": float(diff.max()), "max_ulp": float((diff / np.maximum(spacing, 1.e-45)).max()), "actual": a.tolist()}
    result["precipitation_components"] = {name: value.tolist() for name,value in precip.items()}
    # Original :1299 accumulates the four REAL surface fluxes in this order.
    rain = ((precip["rain"].astype(np.float32) + precip["snow"].astype(np.float32)) + precip["graupel"].astype(np.float32)) + precip["ice"].astype(np.float32)
    ref_rain = np.asarray([r["scalars"]["RAINNCV"] for r in refs], np.float32)
    result["rainncv"] = {"actual": rain.tolist(), "reference": ref_rain.tolist(), "max_abs": float(np.max(np.abs(rain-ref_rain)))}
    result["public_surface"] = surface_projection(precip,refs)
    result["input_census"] = {name: int(np.count_nonzero(a)) for name,a in cols.items() if name.endswith("_IN")}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {k:v for k,v in data.items() if k != "actual"} for name,data in result["fields"].items()}, indent=2))
    print("RAINNCV", result["rainncv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cases", default="1,2,3,4,5,6")
    args = parser.parse_args()
    main(args.reference, args.out, tuple(map(int,args.cases.split(","))))
