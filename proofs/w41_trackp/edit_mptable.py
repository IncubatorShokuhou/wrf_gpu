"""Write an MPTABLE.TBL edited exactly like a Track P multiplier set (forecast-level oracle for original CPU-WRF).

usage: python edit_mptable.py <in MPTABLE.TBL> <out MPTABLE.TBL> <spec.json>
spec: {"z0mvt": {"1": 1.5, ...} | factor, "albsat": factor, "albdry": factor, "rhol": factor, "rhos": factor,
       "eg": factor, "laim": factor | {cat: factor}, "saim": ..., "vcmx25": ..., "mp": ...}
Vegetation rows are edited in the &noahmp_modis_parameters block (the land-use table WN3/Swiss/PROD use), soil
albedo rows (all soil colours; WRF and the port read colour 4) and EG(1) (soil; lake EG(2) untouched) in
&noahmp_rad_parameters. Each edited value is written as repr(float64(value) * float64(factor)), i.e. the value the
port's control expansion rounds once to REAL; Fortran's REAL read of that decimal is the same number (barring
double-rounding ties). Unedited text stays byte-identical.
"""
import json
import re
import sys
from pathlib import Path

ROWS = {"z0mvt": ["Z0MVT"], "rhol": ["RHOL_VIS", "RHOL_NIR"], "rhos": ["RHOS_VIS", "RHOS_NIR"],
        "vcmx25": ["VCMX25"], "mp": ["MP"],
        "laim": [f"LAI_{m}" for m in ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")],
        "saim": [f"SAI_{m}" for m in ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")],
        "albsat": ["ALBSAT_VIS", "ALBSAT_NIR"], "albdry": ["ALBDRY_VIS", "ALBDRY_NIR"], "eg": ["EG"]}
RAD = {"albsat", "albdry", "eg"}


def _block(text, name):
    start = text.index(f"&{name}")
    end = text.index("\n/", start)
    return start, end


def _edit_row(block, key, factor_of):
    pat = re.compile(rf"^(\s*{key}\s*=)([^!\n]*)(.*)$", re.M)
    match = pat.search(block)
    if match is None:
        raise KeyError(key)
    values = [v for v in match.group(2).split(",")]
    out = []
    count = 0
    for raw in values:
        if not raw.strip():
            out.append(raw)
            continue
        count += 1
        factor = factor_of(count)
        if factor == 1.0:
            out.append(raw)
            continue
        new = repr(float(raw.strip()) * float(factor))
        out.append(" " + new)
    return block[:match.start(2)] + ",".join(out) + block[match.end(2):]


def edit(text, spec):
    for name, factor in spec.items():
        block_name = "noahmp_rad_parameters" if name in RAD else "noahmp_modis_parameters"
        start, end = _block(text, block_name)
        block = text[start:end]
        for key in ROWS[name]:
            if name == "eg":
                fn = lambda i, f=factor: float(f) if i == 1 else 1.0  # noqa: E731  EG(1) soil only
            elif isinstance(factor, dict):
                fn = lambda i, f=factor: float(f.get(str(i), f.get(i, 1.0)))  # noqa: E731  1-based VEGTYP
            else:
                fn = lambda i, f=factor: float(f)  # noqa: E731
            block = _edit_row(block, key, fn)
        text = text[:start] + block + text[end:]
    return text


if __name__ == "__main__":
    src, dst, spec_path = sys.argv[1:4]
    spec = json.loads(Path(spec_path).read_text())
    Path(dst).write_text(edit(Path(src).read_text(), spec))
    print(json.dumps({"edited": sorted(spec), "out": dst}))
