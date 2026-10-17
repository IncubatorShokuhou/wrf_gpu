"""Build a lane-private WRF P3 oracle and export REAL4 table/property fixtures.

Run with JAX_PLATFORMS=cpu and a lane-owned --out; no JAX/GPU use here.
Only private visibility is changed in the copied full WRF module. Arithmetic,
initialization, interpolation and particle-property procedures are original.
WRF p3_init's saved is_init requires a fresh process per table configuration.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

N = 128
TABLES = {
    "2mom": {"ice": ("itab", (5, 4, 50, 14)), "rain": ("itabcoll", (5, 4, 50, 30, 2))},
    "3mom": {"ice": ("itab_3mom", (11, 5, 4, 50, 15)),
             "rain": ("itabcoll_3mom", (11, 5, 4, 50, 30, 2))},
    "ice_ice": {"number": ("itabcolli1", (25, 4, 5, 25, 4, 5)),
                "mass": ("itabcolli2", (25, 4, 5, 25, 4, 5))},
    "warm": {"mu": ("mu_r_table", (150,)), "number_speed": ("vn_table", (300, 10)),
             "mass_speed": ("vm_table", (300, 10)), "ventilation": ("revap_table", (300, 10))},
}
OUTPUTS = {"ice_coords": (N, 6), "rain_coords": (N, 2), "mu_coords": (N, 2),
           "pair_coords": (N, 12), "bulk": (N, 3), "mu": (N,), "qzi": (N, 2)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    x = np.zeros((N, 18), dtype=np.float32)
    x[:, 0] = np.geomspace(1.e-13, .02, N)  # qi
    x[:, 1] = np.geomspace(2.e6, 1.e-4, N)  # ni, normalized mass covers both ends
    x[:, 2] = x[:, 0] * np.resize(np.array([0, .09, .34, .66, .93, 1., 1.2]), N)
    x[:, 3] = np.resize([0, 25, 50, 100, 249.9, 250, 450, 649.9, 650, 650.1, 900, 950], N)
    x[:, 4] = np.geomspace(1.e-16, .01, N)  # rain mass
    x[:, 5] = np.geomspace(1.e7, 1.e-3, N)  # rain number
    x[::13, 5] = 0
    x[:, 6] = np.linspace(-1, 24, N)  # mu_i
    x[:, 7] = x[::-1, 0] * .73  # second category
    x[:, 8] = x[::-1, 1] * 1.31
    x[:, 9] = x[:, 7] * np.resize([0, .21, .51, .81, 1.], N)
    x[:, 10] = x[:, 2] / np.maximum(x[:, 3], 1)  # rime volume1
    x[:, 11] = x[:, 9] / np.maximum(x[::-1, 3], 1)  # volume2
    x[::7, 10] = 1.e-16  # BETWEEN collector bsmall and collectee 1e-15
    x[::7, 11] = 1.e-16
    x[1, 2], x[1, 10] = 1.e-15, 1.e-15  # terminal qsmall clears mass/volume only
    x[:, 12] = 1.  # mom0
    x[:, 13] = 1.e-5  # mom3
    g = np.resize([.9, 1.229, 1.230, 1.404, 1.405, 1.792, 1.793, 2.945,
                   2.946, 4.199, 4.2, 7.122, 7.123, 13.309, 13.31, 19.999, 20., 25.], N)
    x[:, 14] = g * 1.e-10  # mom6
    x[:, 15] = np.geomspace(1.e-30, 1.e-3, N)  # advected sqrt(N*Z)
    x[:, 16] = np.geomspace(1.e-32, 1.e-10, N)  # raw Z
    x[:, 17] = np.geomspace(1.e-16, 1.e7, N)  # number for QZI threshold
    return x


def driver(kind):
    three = kind == "3mom"
    nchan = 15 if three else 14
    tables = dict(TABLES[kind])
    if kind == "2mom":
        tables.update(TABLES["warm"])
    writes = "\n".join(
        f"open(newunit=u,file='table_{name}.bin',access='stream',form='unformatted',status='replace'); "
        f"write(u) {symbol}; close(u)" for name, (symbol, _) in tables.items())
    if kind == "ice_ice":
        lookup = """
call access_lookup_table_colli(jjc,iic,ic,jj,ii,i,1,ac,bc,cc,a,b,c,look(r,1))
call access_lookup_table_colli(jjc,iic,ic,jj,ii,i,2,ac,bc,cc,a,b,c,look(r,2))
"""
    elif three:
        lookup = """
do h=1,15
call access_lookup_table_3mom(iz,jj,ii,i,h,a,b,c,z,look(r,h))
enddo
do h=1,2
call access_lookup_table_coll_3mom(iz,jj,ii,j,i,h,a,d,b,c,z,collect(r,h))
enddo
"""
    else:
        lookup = """
do h=1,14
call access_lookup_table(jj,ii,i,h,a,b,c,look(r,h))
enddo
do h=1,2
call access_lookup_table_coll(jj,ii,j,i,h,a,d,b,c,collect(r,h))
enddo
"""
    return f"""program oracle
use microphy_p3
implicit none
integer,parameter :: n={N}, nch={2 if kind == 'ice_ice' else nchan}
integer :: r,u,i,ii,jj,j,iz,ic,iic,jjc,h,stat
real :: x(n,18),coords(n,6),rcoords(n,2),zcoords(n,2),pairs(n,12)
real :: bulk(n,3),mus(n),qzis(n,2),look(n,nch),collect(n,2)
real :: a,b,c,d,z,ac,bc,cc,qrim,brim,rho
character(len=1024) :: tabledir
call get_command_argument(1,tabledir)
call p3_init(trim(tabledir),{2 if kind == 'ice_ice' else 1},{'.true.' if three else '.false.'}, &
model='WRF',stat=stat,abort_on_err=.true.)
if (stat/=STATUS_OK) stop 2
{writes}
open(newunit=u,file='inputs.bin',access='stream',form='unformatted',status='old')
read(u) x
close(u)
do r=1,n
call find_lookupTable_indices_1a(i,jj,ii,a,b,c,50,4,5,x(r,1),x(r,2),x(r,3),x(r,4))
coords(r,:)=[real(i),real(ii),real(jj),a,b,c]
call find_lookupTable_indices_1b(j,d,30,x(r,5),x(r,6))
rcoords(r,:)=[real(j),d]
call find_lookupTable_indices_1c(iz,z,11,x(r,1),x(r,7))
zcoords(r,:)=[real(iz),z]
if ({'.true.' if kind == 'ice_ice' else '.false.'}) then
call find_lookupTable_indices_2(i,ii,jj,ic,iic,jjc,a,b,c,ac,bc,cc,25,4,5, &
x(r,1),x(r,8),x(r,2),x(r,9),x(r,3),x(r,10),x(r,11),x(r,12))
pairs(r,:)=[real(i),real(ii),real(jj),a,b,c,real(ic),real(iic),real(jjc),ac,bc,cc]
endif
{lookup}
qrim=x(r,3); brim=x(r,11)
call calc_bulkRhoRime(x(r,1),qrim,brim,rho)
bulk(r,:)=[qrim,brim,rho]
mus(r)=compute_mu_3moment(x(r,13),x(r,14),x(r,15),20.)
if (x(r,18)>=1.e-14) then
qzis(r,1)=x(r,16)**2/x(r,18)
else
qzis(r,1)=0.
endif
if (x(r,18)>=1.e-14.and.x(r,17)>=1.e-30) then
qzis(r,2)=(x(r,17)*x(r,18))**0.5
else
qzis(r,2)=0.
endif
enddo
open(newunit=u,file='ice_coords.bin',access='stream',form='unformatted',status='replace'); write(u) coords; close(u)
open(newunit=u,file='rain_coords.bin',access='stream',form='unformatted',status='replace'); write(u) rcoords; close(u)
open(newunit=u,file='mu_coords.bin',access='stream',form='unformatted',status='replace'); write(u) zcoords; close(u)
open(newunit=u,file='bulk.bin',access='stream',form='unformatted',status='replace'); write(u) bulk; close(u)
open(newunit=u,file='mu.bin',access='stream',form='unformatted',status='replace'); write(u) mus; close(u)
open(newunit=u,file='qzi.bin',access='stream',form='unformatted',status='replace'); write(u) qzis; close(u)
open(newunit=u,file='lookup.bin',access='stream',form='unformatted',status='replace'); write(u) look; close(u)
{'open(newunit=u,file="pair_coords.bin",access="stream",form="unformatted",status="replace"); write(u) pairs; close(u)' if kind == 'ice_ice' else 'open(newunit=u,file="collection.bin",access="stream",form="unformatted",status="replace"); write(u) collect; close(u)'}
end program
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--kinds", nargs="+", choices=("2mom", "3mom", "ice_ice"), default=("2mom", "3mom", "ice_ice"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.wrf_root = args.wrf_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.wrf_root / "phys/module_mp_p3.F"
    text = source.read_text()
    exports = ["find_lookupTable_indices_1a", "find_lookupTable_indices_1b", "find_lookupTable_indices_1c",
               "find_lookupTable_indices_2", "calc_bulkRhoRime", "compute_mu_3moment",
               "access_lookup_table", "access_lookup_table_coll", "access_lookup_table_colli",
               "access_lookup_table_3mom", "access_lookup_table_coll_3mom"]
    exports += [symbol for group in TABLES.values() for symbol, _ in group.values()]
    exposure = "\n".join(f" public :: {name}" for name in exports)
    staged, count = re.subn(r"(?m)^ contains$", exposure + "\n contains", text, count=1)
    assert count == 1
    assert staged.replace(exposure + "\n contains", " contains", 1) == text
    (args.out / "module_mp_p3.F").write_text(staged)
    flags = ["-cpp", "-ffree-form", "-ffree-line-length-none", "-O2", "-ffp-contract=off"]
    subprocess.run([args.compiler, *flags, "-c", "module_mp_p3.F"], cwd=args.out, check=True, timeout=120)
    asset_dir = args.out / "assets"
    fixture_dir = args.out / "fixtures"
    asset_dir.mkdir(exist_ok=True)
    fixture_dir.mkdir(exist_ok=True)
    manifest = {"fixture_id": "p3-real4-lookup-properties-v1", "source": str(source),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.wrf_root).decode().strip(),
                "source_sha256": sha(source), "staged_sha256": sha(args.out / "module_mp_p3.F"),
                "modifications": "private visibility only; no computational statement changed",
                "compiler": subprocess.check_output([args.compiler, "--version"]).decode().splitlines()[0],
                "flags": flags, "precision": "REAL4; original explicit DOUBLE table-read intermediates",
                "scenario": "128 positive-ice/rime/rain/2-category/3-moment property samples; not a forecast",
                "tolerances": {"integer_indices": "exact", "coordinates": {"rtol": 8.e-6, "atol": 2.e-6},
                               "lookup": {"rtol": 4.e-6, "atol": 1.e-30}, "bulk_qzi_eager": "exact",
                               "mu": {"rtol": 4.e-6, "atol": 2.e-5}},
                "files": {}, "raw_tables": {}, "license_notes": "WRF P3 source/table derivation; see repository THIRD_PARTY_NOTICES.md"}
    previous = args.out / "manifest.json"
    if previous.is_file():
        previous = json.loads(previous.read_text())
        assert previous["source_sha256"] == manifest["source_sha256"]
        for name, item in previous["files"].items():
            assert sha(args.out / name) == item["sha256"], name
        manifest["files"] = previous["files"]
    for path in (args.wrf_root / "run").glob("p3_lookupTable*.dat*"):
        manifest["raw_tables"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    for kind in args.kinds:
        work = args.out / kind
        work.mkdir(exist_ok=True)
        x = inputs()
        (work / "inputs.bin").write_bytes(x.tobytes(order="F"))
        code = work / "oracle.f90"
        code.write_text(driver(kind))
        subprocess.run([args.compiler, *flags, "-I", str(args.out.resolve()), str(code.resolve()),
                        str((args.out / "module_mp_p3.o").resolve()), "-o", str((work / "oracle").resolve())], check=True, timeout=120)
        with (work / "stdout.log").open("w") as log:
            subprocess.run([str((work / "oracle").resolve()), str((args.wrf_root / "run").resolve())],
                           cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        groups = [kind, "warm"] if kind == "2mom" else [kind]
        for group in groups:
            arrays = {key: np.fromfile(work / f"table_{key}.bin", dtype=np.float32).reshape(shape, order="F")
                      for key, (_, shape) in TABLES[group].items()}
            assert all(np.isfinite(a).all() for a in arrays.values())
            np.savez_compressed(asset_dir / f"{group}.npz", **arrays)
            manifest["files"][f"assets/{group}.npz"] = {"sha256": sha(asset_dir / f"{group}.npz"),
                "variables": {k: {"shape": list(v.shape), "dtype": "float32"} for k, v in arrays.items()}}
        shapes = dict(OUTPUTS)
        shapes.pop("pair_coords")
        shapes["lookup"] = (N, 2 if kind == "ice_ice" else (15 if kind == "3mom" else 14))
        shapes["pair_coords" if kind == "ice_ice" else "collection"] = (N, 12 if kind == "ice_ice" else 2)
        fixture = {name: np.fromfile(work / f"{name}.bin", dtype=np.float32).reshape(shape, order="F")
                   for name, shape in shapes.items()}
        fixture["inputs"] = x
        assert all(np.isfinite(a).all() for a in fixture.values())
        np.savez_compressed(fixture_dir / f"{kind}.npz", **fixture)
        manifest["files"][f"fixtures/{kind}.npz"] = {"sha256": sha(fixture_dir / f"{kind}.npz"),
            "variables": {k: {"shape": list(v.shape), "dtype": "float32"} for k, v in fixture.items()}}
    manifest["census"] = {"samples": N, "positive_qi": int((x[:, 0] > 0).sum()),
                          "positive_qir": int((x[:, 2] > 0).sum()), "positive_qi2": int((x[:, 7] > 0).sum()),
                          "positive_qir2": int((x[:, 9] > 0).sum()), "collector_collectee_threshold_gap": int(((x[:, 10] > 1.e-17) & (x[:, 10] < 1.e-15)).sum())}
    enrich_manifest(manifest)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "manifest.yaml").write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in manifest.items()))
    print(json.dumps({"source": manifest["source_sha256"], "census": manifest["census"],
                      "asset_bytes": sum(p.stat().st_size for p in asset_dir.glob("*.npz"))}, indent=2))


def enrich_manifest(manifest):
    """Metadata-only updates can reuse already measured numerical fixtures."""
    manifest["variables"] = {
        "inputs": {"names": ["qi", "ni", "qir", "rho_rime", "qr", "nr", "mu_i", "qi2", "ni2",
                              "qir2", "bir1", "bir2", "mom0", "mom3", "mom6", "advected_qzi", "raw_z", "qzi_ni"],
                   "units": ["kg kg-1", "kg-1", "kg kg-1", "kg m-3", "kg kg-1", "kg-1", "1",
                             "kg kg-1", "kg-1", "kg kg-1", "m3 kg-1", "m3 kg-1", "m-3", "1", "m3",
                             "m3 kg-1", "m6 kg-1", "kg-1"], "shape": [N, 18], "dtype": "float32"},
        "ice_coords": {"units": "1 (ONE-BASED mass,rime,density indices then coordinates)", "shape": [N, 6], "dtype": "float32"},
        "rain_coords": {"units": "1 (ONE-BASED index,coordinate)", "shape": [N, 2], "dtype": "float32"},
        "mu_coords": {"units": "1 (ONE-BASED index,coordinate)", "shape": [N, 2], "dtype": "float32"},
        "pair_coords": {"units": "1 (collector six then collectee six)", "shape": [N, 12], "dtype": "float32"},
        "bulk": {"units": ["kg kg-1", "m3 kg-1", "kg m-3"], "shape": [N, 3], "dtype": "float32"},
        "mu": {"units": "1", "shape": [N], "dtype": "float32"},
        "qzi": {"units": ["m6 kg-1", "m3 kg-1"], "shape": [N, 2], "dtype": "float32",
                "status": "original WRF evidence ONLY; conversion port withheld after FTZ mismatch"},
        "lookup": {"units": "original channel units; unscaled WRF itab/itab_3mom/itabcolli outputs",
                   "shape": "[128,14]/[128,15]/[128,2] per variant", "dtype": "float32"},
        "collection": {"units": "original WRF ice-rain table log10 channel values",
                       "shape": [N, 2], "dtype": "float32"},
    }
    for name, variable in manifest["variables"].items():
        tolerance = ("integer_indices" if name.endswith("coords") else
                     "bulk_qzi_eager" if name in ("bulk", "qzi") else "mu" if name == "mu" else "lookup")
        variable["tolerance"] = manifest["tolerances"][tolerance]


if __name__ == "__main__":
    main()
