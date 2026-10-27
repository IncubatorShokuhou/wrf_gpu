"""Extract reusable Nt_c tables without loading whole Fortran records/assets.

Pristine WRF freezeH2O generates every t_Nc bin independently of Nt_c. Keep
all 100 bins at the non-aerosol IN index 28 (Fortran), preserving DOUBLE data.
Verify REAL gamma values against both the retained bit table and a fresh tiny
original-Fortran constants export. No gpuwrf/JAX imports or model/GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

import numpy as np


FREEZE_SHA256 = "05497b0ed3f8dee78c9db25374f3e7f3da891580b15974453b608631433cce46"
N_C, N_NC, N_T, N_IN = 37, 100, 45, 55
FIXED_IN = 27  # zero based; WRF mp8 fixed-IN index 28.


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def function(source, name):
    match = re.search(rf"^\s*REAL FUNCTION {name}\([^\n]*\n.*?^\s*END FUNCTION {name}\s*$",
                      source, re.M | re.S | re.I)
    assert match, name
    return match[0]


def canonical_fortran(source):
    return re.sub(r"\s+", "", "\n".join(line.split("!")[0] for line in source.splitlines())).lower()


def wrf_real_constants(wrf_source, gamma_source, gamma_table, compiler, directory):
    """Compile only verbatim original gamma/axis formulas, not the WRF model."""
    text, witness = wrf_source.read_text(), gamma_source.read_text()
    functions = []
    for name in ("GAMMLN", "WGAMMA"):
        original = function(text, name)
        assert canonical_fortran(original) == canonical_fortran(function(witness, name)), name
        functions.append(original)
    gamma = re.search(r"      do n = 1, 15\n.*?      enddo", text[text.index("!..These constants various exponents"):], re.S)[0]
    axis = re.search(r"      xDx\(1\) = 1.0d0\n.*?      nic1 = DLOG\(t_Nc\(nbc\)/t_Nc\(1\)\)",
                     text[text.index("!..Create bins of cloud droplet number concentration"):], re.S)[0]
    assert not re.search(r"\bNt_c\b|\bmu_c\b", gamma + axis, re.I)
    assert re.search(r"REAL, PARAMETER, PRIVATE:: bm_r\s*=\s*3\.0", text, re.I)
    assert re.search(r"REAL, PARAMETER, PRIVATE:: bv_c\s*=\s*2\.0", text, re.I)
    for declaration in ("PI = 3.1415926536", "rho_w = 1000.0", "am_r = PI*rho_w/6.0", "D0r = 50.E-6"):
        assert declaration in text, declaration
    assert "x = (ccg(3)*ocg2)**obmr" in gamma_source.with_name("wrf_constants.F90").read_text()
    assert "x = am_r*nu_c*10.*D0r*D0r*D0r" in gamma_source.with_name("wrf_constants.F90").read_text()
    driver = "\n".join(("MODULE ntc_wrf_probe", "CONTAINS", *functions, "END MODULE ntc_wrf_probe",
        "PROGRAM export_constants", "USE ntc_wrf_probe", "IMPLICIT NONE",
        "INTEGER, PARAMETER :: nbc=100", "INTEGER :: n, nic1, count, nu_out, idx_out",
        "REAL, PARAMETER :: bm_r=3.0, bv_c=2.0",
        "REAL, PARAMETER :: PI=3.1415926536, rho_w=1000.0, am_r=PI*rho_w/6.0, D0r=50.E-6",
        "REAL :: cce(5,15), ccg(5,15), ocg1(15), ocg2(15)",
        "REAL :: dc_g_prefix(15), pnr_wau_den(15), obmr, mu_out", "REAL, ALLOCATABLE :: nc_input(:)",
        "DOUBLE PRECISION :: xDx(nbc+1), t_Nc(nbc)", gamma, axis,
        "obmr=1./bm_r", "DO n=1,15",
        "dc_g_prefix(n)=(ccg(3,n)*ocg2(n))**obmr",
        "pnr_wau_den(n)=am_r*n*10.*D0r*D0r*D0r", "END DO",
        "OPEN(10,FILE='constants.bin',ACCESS='stream',FORM='unformatted',STATUS='new')",
        "WRITE(10) cce, ccg, ocg1, ocg2, t_Nc, nic1, dc_g_prefix, pnr_wau_den", "CLOSE(10)",
        "OPEN(11,FILE='probe_input.bin',ACCESS='stream',FORM='unformatted',STATUS='old')",
        "READ(11) count", "ALLOCATE(nc_input(count))", "READ(11) nc_input", "CLOSE(11)",
        "OPEN(12,FILE='probe_output.bin',ACCESS='stream',FORM='unformatted',STATUS='new')",
        "DO n=1,count", "nu_out=MIN(15,NINT(1000.E6/nc_input(n))+2)",
        "mu_out=MIN(15.,1000.E6/nc_input(n)+2.)",
        "idx_out=NINT(1.0+FLOAT(nbc)*DLOG(nc_input(n)/t_Nc(1))/nic1)",
        "idx_out=MAX(1,MIN(idx_out,nbc))", "WRITE(12) nu_out, mu_out, idx_out", "END DO", "CLOSE(12)",
        "END PROGRAM export_constants"))
    directory.mkdir()
    (directory / "export.f90").write_text(driver + "\n")
    sample_nc = [2., 50.e6, 100.e6, 150.e6, 300.e6, 1999.e6]
    t_nc1_guess = math.sqrt(math.exp(math.log(3000.) / 100.)) * 1.e6
    boundaries = [1.e9 / (n + .5) for n in range(1, 13)]
    boundaries += [t_nc1_guess * math.exp((i - 1.5) * 7. / 100.) for i in (2, 66, 82, 100)]
    for boundary in boundaries:
        center = np.float32(boundary)
        sample_nc += [np.nextafter(center, np.float32(-np.inf)), center, np.nextafter(center, np.float32(np.inf))]
    sample_nc = np.unique(np.asarray(sample_nc, np.float32))
    with (directory / "probe_input.bin").open("wb") as stream:
        stream.write(struct.pack(">i", len(sample_nc)))
        stream.write(sample_nc.astype(">f4").tobytes())
    flags = ["-w", "-ffree-form", "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4",
             "-fallow-argument-mismatch", "-fallow-invalid-boz", "-O2", "-ftree-vectorize", "-funroll-loops"]
    command = [str(compiler), *flags, "export.f90", "-o", "export"]
    try:
        with (directory / "build.log").open("w") as log:
            subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            subprocess.run([str(directory / "export")], cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=15)
        arrays = {}
        with (directory / "constants.bin").open("rb") as stream:
            for name, shape, dtype in (("cce", (5, 15), ">f4"), ("ccg", (5, 15), ">f4"),
                                       ("ocg1", (15,), ">f4"), ("ocg2", (15,), ">f4"),
                                       ("t_Nc", (100,), ">f8"), ("nic1", (), ">i4"),
                                       ("dc_g_prefix", (15,), ">f4"), ("pnr_wau_den", (15,), ">f4")):
                count = int(np.prod(shape))
                raw = stream.read(count * np.dtype(dtype).itemsize)
                arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F").astype(np.dtype(dtype).newbyteorder("="))
            assert not stream.read(1)
        assert all(np.isfinite(value).all() for value in arrays.values())
        bits = np.zeros((5, 15), np.uint32)
        seen = set()
        for row in gamma_table.read_text().splitlines():
            match = re.fullmatch(r"ccg\((\d+),(\d+)\) arg\s+(\S+) val\s+(\S+)\s+(\d+)", row)
            if not match:
                continue
            family, nu = int(match[1]), int(match[2])
            assert (family, nu) not in seen
            seen.add((family, nu))
            assert float(match[3]) == nu + (1, 4, 7, 3, 6)[family - 1]
            value = np.float32(float(match[4]))
            assert value.view(np.uint32).item() == int(match[5])
            bits[family - 1, nu - 1] = int(match[5])
        assert len(seen) == 75 and np.array_equal(arrays["ccg"].view(np.uint32), bits)
        for name, row in (("ocg1", 0), ("ocg2", 1)):
            expected = np.divide(np.float32(1), arrays["ccg"][row], dtype=np.float32)
            assert arrays[name].tobytes() == expected.tobytes(), name
        assert int(arrays["nic1"]) == 7
        assert int(np.log(arrays["t_Nc"][-1] / arrays["t_Nc"][0])) == 7
        assert arrays["dc_g_prefix"][11] == np.float32(16.98038101196289)
        assert arrays["pnr_wau_den"][11] == np.float32(7.853981465189008e-09)
        probe = np.fromfile(directory / "probe_output.bin", dtype=[("nu", ">i4"), ("mu", ">f4"), ("idx", ">i4")])
        assert len(probe) == len(sample_nc)
        ratio = np.divide(np.float32(1.e9), sample_nc, dtype=np.float32)
        expected_nu = np.minimum(15, np.floor(ratio.astype(np.float64) + .5).astype(np.int32) + 2)
        expected_mu = np.minimum(np.float32(15), ratio + np.float32(2))
        expression = 1. + 100. * np.log(sample_nc.astype(np.float64) / arrays["t_Nc"][0]) / int(arrays["nic1"])
        expected_idx = np.clip(np.floor(expression + .5).astype(np.int32), 1, 100)
        assert np.array_equal(probe["nu"], expected_nu)
        assert np.array_equal(probe["mu"], expected_mu)
        assert np.array_equal(probe["idx"], expected_idx)
        probe_rows = [dict(nt_c=float(nc), nt_c_bits=int(nc.view(np.uint32)), nu_c=int(row["nu"]),
                           mu_c=float(row["mu"]), idx_n=int(row["idx"])) for nc, row in zip(sample_nc, probe, strict=True)]
        (directory / "index_probe.json").write_text(json.dumps(probe_rows, indent=2) + "\n")
        return arrays, dict(command=command, original_formulas_sha256=hashlib.sha256((gamma + axis).encode()).hexdigest(),
                            driver_sha256=sha(directory / "export.f90"), gamma_bit_matches=75,
                            reciprocal_bit_matches=30, prefix_default_bit_matches=2,
                            index_probe_samples=len(sample_nc), index_probe_sha256=sha(directory / "index_probe.json"),
                            binary_sha256=sha(directory / "constants.bin"))
    finally:
        for name in ("export", "ntc_wrf_probe.mod", "constants.bin", "probe_input.bin", "probe_output.bin"):
            (directory / name).unlink(missing_ok=True)


def qcfz_planes(path):
    """Seek across the four rain records; read only one IN plane per cloud record."""
    expected = [37 * 37 * N_T * N_IN * 8] * 4 + [N_C * N_NC * N_T * N_IN * 8] * 2
    plane_bytes = N_C * N_NC * N_T * 8
    arrays, rows = {}, []
    with path.open("rb") as stream:
        for index, size in enumerate(expected):
            start = stream.tell()
            marker = stream.read(4)
            assert len(marker) == 4 and struct.unpack(">i", marker)[0] == size, (index, start)
            if index >= 4:
                stream.seek(start + 4 + FIXED_IN * plane_bytes)
                raw = stream.read(plane_bytes)
                assert len(raw) == plane_bytes
                name = ("tpi_qcfz", "tni_qcfz")[index - 4]
                arrays[name] = np.ascontiguousarray(np.frombuffer(raw, ">f8").reshape((N_C, N_NC, N_T), order="F"), dtype=np.float64)
                assert np.isfinite(arrays[name]).all()
            stream.seek(start + 4 + size)
            assert struct.unpack(">i", stream.read(4))[0] == size
            rows.append(dict(record=index + 1, offset=start, bytes=size, payload_bytes_read=plane_bytes if index >= 4 else 0))
        assert not stream.read(1), "unexpected trailing records"
    return arrays, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--wrf-source", type=Path, required=True)
    parser.add_argument("--gamma-source", type=Path, required=True)
    parser.add_argument("--gamma-table", type=Path, required=True)
    parser.add_argument("--mixed-asset", type=Path, required=True)
    parser.add_argument("--aero-asset", type=Path)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix(".json").exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    freeze_hash = sha(args.freeze)  # Streamed hash; never read_bytes on the 255 MB file.
    assert freeze_hash == FREEZE_SHA256, freeze_hash
    constants, constants_receipt = wrf_real_constants(args.wrf_source, args.gamma_source, args.gamma_table,
                                                      args.compiler, args.out.parent / "wrf_constants")
    planes, records = qcfz_planes(args.freeze)
    nc_default = np.float64(np.float32(100.e6))
    index = int(np.floor(1. + N_NC * np.log(nc_default / constants["t_Nc"][0]) / int(constants["nic1"]) + .5))
    assert index == 66
    with np.load(args.mixed_asset, allow_pickle=False) as old:
        for name, plane in planes.items():
            assert plane[:, index - 1, :].tobytes() == old[name].tobytes(), name
    aero = {}
    if args.aero_asset:
        # Deliberately do not read the full aero file, qcfz, or evaporation arrays.
        with np.load(args.aero_asset, allow_pickle=False) as old:
            for name in ("ccg", "ocg1", "ocg2", "t_Nc", "nic1"):
                value = np.asarray(old[name])
                reference = constants[name]
                rounded = value.astype(reference.dtype)
                aero[name] = dict(shape=list(value.shape), stored_dtype=str(value.dtype),
                                  stored_value_differences=int(np.count_nonzero(value != reference)),
                                  rounded_bitwise_elements=int(np.count_nonzero(
                                      rounded.reshape(-1).view(f"u{reference.dtype.itemsize}") !=
                                      reference.reshape(-1).view(f"u{reference.dtype.itemsize}"))),
                                  max_abs=float(np.max(np.abs(value - reference))),
                                  selected_member_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    arrays = {**planes, **constants, "fixed_in_index": np.array(FIXED_IN, np.int32)}
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    receipt = dict(scope="Nt_c-independent WRF table export, not a model/fidelity/GPU gate", platform="cpu", gpu_executed=False,
                   tool_sha256=sha(Path(__file__)),
                   freeze_path=str(args.freeze), freeze_sha256=freeze_hash, records=records,
                   qcfz_payload_bytes_read=2 * N_C * N_NC * N_T * 8,
                   fixed_in_index_1based=FIXED_IN + 1, default_nc_index_1based=index, default_slices_byte_exact=True,
                   original_source_sha256=sha(args.wrf_source), gamma_source_sha256=sha(args.gamma_source),
                   gamma_table_sha256=sha(args.gamma_table), mixed_asset_sha256=sha(args.mixed_asset),
                   constants=constants_receipt, t_nc1=float(constants["t_Nc"][0]),
                   t_nc1_bits=int(constants["t_Nc"][0].view(np.uint64)), nic1=int(constants["nic1"]),
                   default_dc_g_prefix=float(constants["dc_g_prefix"][11]),
                   default_pnr_wau_den=float(constants["pnr_wau_den"][11]),
                   aero_selected_members=aero, output_sha256=sha(args.out), output_bytes=args.out.stat().st_size,
                   arrays={name:dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                                    sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name,value in arrays.items()})
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
