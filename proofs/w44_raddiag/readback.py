"""Compiled trajectory byte gate and opt-in field readback; no GPU-vs-GPU fidelity claim."""
import argparse
import json
from pathlib import Path
import tempfile

from netCDF4 import Dataset
import numpy as np

CORE = ("U", "V", "W", "T", "THM", "P", "PH", "MU", "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP")
RAD = ("RTHRATLW", "RTHRATSW", "RTHRATLWC", "RTHRATSWC", "QC_RAD", "QI_RAD", "QS_RAD", "QG_RAD")


def compare(off, on, *, frames=12):
    left = {p.name: p for p in Path(off).glob("wrfout_d0*")}
    right = {p.name: p for p in Path(on).glob("wrfout_d0*")}
    if set(left) != set(right) or len(left) != frames:
        raise ValueError(f"incomplete frame coverage: {len(left)}/{len(right)}, expected {frames}")
    rows = []
    for name in sorted(left):
        with Dataset(left[name]) as a, Dataset(right[name]) as b:
            a.set_auto_mask(False); b.set_auto_mask(False)
            if not set(CORE + ("QI_BL", "CLDFRA")).issubset(a.variables):
                raise ValueError(f"missing prognostic/history coverage: {name}")
            if set(RAD) & set(a.variables) or not set(RAD).issubset(b.variables):
                raise ValueError("rad_diag OFF/ON field selection is wrong")
            diffs = []
            common_diffs = []
            for field in sorted(set(a.variables) & set(b.variables)):
                x, y = np.asarray(a[field][:]), np.asarray(b[field][:])
                equal = (x.shape, x.dtype, x.tobytes()) == (y.shape, y.dtype, y.tobytes())
                if not equal:
                    common_diffs.append(field)
                if field in CORE:
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError(f"nonfinite prognostic {name}:{field}")
                    if not equal:
                        diffs.append(dict(field=field, unequal_elements=int(np.count_nonzero(x != y)),
                                          max_abs=float(np.max(np.abs(x.astype(np.float64)-y)))))
            for field in RAD:
                x = np.asarray(b[field][:])
                if x.shape != b["T"].shape or x.dtype != np.float32 or not np.isfinite(x).all():
                    raise ValueError(f"invalid rad_diag field {name}:{field}")
            rows.append(dict(frame=name, prognostic_differences=diffs, common_different=common_diffs,
                             rad_fields=list(RAD)))
    return dict(prognostics_byte_identical=all(not r["prognostic_differences"] for r in rows),
                frames=len(rows), rows=rows, evidence="GPU-vs-GPU compiled regression only")


def selftest(directory):
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        root = Path(temp)
        for arm in ("off", "on"):
            (root/arm).mkdir()
            with Dataset(root/arm/"wrfout_d01_test", "w") as ds:
                ds.createDimension("n", 2)
                for name in (*CORE, "QI_BL", "CLDFRA", *(RAD if arm == "on" else ())):
                    ds.createVariable(name, "f4", ("n",))[:] = [1, 2]
        assert compare(root/"off", root/"on", frames=1)["prognostics_byte_identical"]
        with Dataset(root/"on"/"wrfout_d01_test", "a") as ds:
            ds["U"][0] = np.nextafter(np.float32(1), np.float32(2))
        assert not compare(root/"off", root/"on", frames=1)["prognostics_byte_identical"]
        with Dataset(root/"on"/"wrfout_d01_test", "a") as ds:
            ds["U"][0] = np.nan
        try:
            compare(root/"off", root/"on", frames=1)
        except ValueError as error:
            assert "nonfinite" in str(error)
        else:
            raise AssertionError("NaN mutation survived")
    print("readback dry test PASS: baseline, 1ULP, NaN")


def fields_only(directory):
    files = sorted(Path(directory).glob("wrfout_d0*"))
    if len(files) != 12:
        raise ValueError(f"incomplete 3-domain/4-frame readback: {len(files)}")
    rows = []
    for path in files:
        with Dataset(path) as ds:
            for name in (*RAD, "QI_BL", "CLDFRA"):
                value = np.asarray(ds[name][:])
                if value.shape != ds["T"].shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid held field {path}:{name}")
            rows.append(dict(frame=path.name, fields=[*RAD, "QI_BL", "CLDFRA"]))
    return dict(frames=len(rows), rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("off", nargs="?")
    parser.add_argument("on", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--selftest")
    parser.add_argument("--fields-only")
    args = parser.parse_args()
    if args.selftest:
        selftest(args.selftest)
    elif args.fields_only:
        result = fields_only(args.fields_only)
        Path(args.off).write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = compare(args.off, args.on)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(0 if result["prognostics_byte_identical"] else 42)
