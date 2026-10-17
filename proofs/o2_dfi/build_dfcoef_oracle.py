"""Build the pristine-WRF DFI filter-coefficient oracle (tracked fixture dfcoef_oracle.json).

The routines dfcoef .. cnr are cut VERBATIM (byte range by subroutine markers) from the pristine
WRF 4.7.1 share/dfi.F (sha256 pinned below), compiled with the CPU-WRF oracle build's compiler and
optimisation/ISA flags (gfortran -O2 -march=nocona, i.e. no FMA, like
<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/_build_gen2_dmpar/CMakeCache.txt) together with
dfcoef_driver.F90 (a literal copy of optfil_driver's arithmetic), and run for a configuration list.
Every hcoeff is recorded as its exact binary32 bit pattern.

usage: python build_dfcoef_oracle.py <workdir> [<out.json>]
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DFI_F = Path("<USER_HOME>/src/wrf_pristine/WRF/share/dfi.F")
DFI_SHA = "c7b94989ccdf637ebe756b9dcdff26d5b9ba798852ec74c3415a853a32a4ae0f"
FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
FLAGS = ["-cpp", "-ffree-form", "-O2", "-march=nocona", "-ffree-line-length-none"]
HERE = Path(__file__).resolve().parent
# (nfilter, rundfi_s, dt_s, tauc_s): nfilter 0..7 (8 = rhofil is out-of-bounds in WRF itself and
# -1 = debug reads uninitialised memory -> both refused by the port), backward (dt<0) and forward
# windows, odd and even 1+rundfi/dt parities, non-divisible windows, several dt/cutoffs.
CONFIGS = [
    (nf, run, dt, tauc)
    for nf in range(0, 8)
    for run, dt, tauc in (
        (1080, -18.0, 1200.0), (1080, 18.0, 1200.0),     # Swiss oracle A (60 steps each stage)
        (1200, -18.0, 1200.0), (1200, 18.0, 1200.0),     # HRRR-like 20 min, non-divisible
        (3600, -54.0, 3600.0), (1800, 54.0, 3600.0),     # PROD dt, WRF user-guide 1 h window
        (600, -6.0, 900.0), (360, 18.0, 600.0),          # even/odd parities
    )
]


def main() -> None:
    work = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "dfcoef_oracle.json"
    raw = DFI_F.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DFI_SHA, "pristine dfi.F changed"
    text = raw.decode()
    start = text.index("   SUBROUTINE dfcoef (NSTEPS,DT,TAUC,NFILT,H)")
    end = text.index("END FUNCTION cnr", start) + len("END FUNCTION cnr")
    work.mkdir(parents=True, exist_ok=True)
    (work / "dfi_filters.F").write_text(text[start:end] + "\n")
    exe = work / "dfcoef_oracle.exe"
    subprocess.run([FC, *FLAGS, "-o", str(exe), str(HERE / "dfcoef_driver.F90"),
                    str(work / "dfi_filters.F")], check=True, cwd=work)
    stdin = "".join(f"{nf} {run} {dt!r} {tauc!r}\n" for nf, run, dt, tauc in CONFIGS)
    res = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True, cwd=work)
    cases, cur = [], None
    for line in res.stdout.splitlines():
        if line.startswith("CFG "):
            nf, run, dt_hex, tauc_hex, nout = line.split()[1:]
            cur = {"nfilter": int(nf), "rundfi_s": int(run), "dt_bits": dt_hex.lower(),
                   "tauc_bits": tauc_hex.lower(), "hcoeff_bits": []}
            cases.append(cur)
        elif cur is not None and re.fullmatch(r"[0-9A-Fa-f]{8}", line.strip()):
            # DOLPHWIN also WRITEs its diagnostics to stdout; only bit patterns are weights.
            cur["hcoeff_bits"].append(line.strip().lower())
    assert len(cases) == len(CONFIGS)
    for case, (_nf, _run, dt, tauc) in zip(cases, CONFIGS):
        case["dt_s"], case["tauc_s"] = dt, tauc
    payload = {
        "schema": "o2_dfi.dfcoef_oracle.v1",
        "source": str(DFI_F), "source_sha256": DFI_SHA, "wrf_version": "4.7.1",
        "routines": "dfcoef normlz debug uniform lanczos hamming blackman kaiser bessi0 potter2 dolphwin dolph cheby"
                    " rhofil rhocof cnr (verbatim); driver = optfil_driver arithmetic",
        "compiler": FC, "flags": FLAGS,
        "driver_sha256": hashlib.sha256((HERE / "dfcoef_driver.F90").read_bytes()).hexdigest(),
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
