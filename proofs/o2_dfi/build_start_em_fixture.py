"""Crop the CPU-WRF TDFI oracle's filtered state into tests/fixtures/o2_dfi_start_em_crop.npz.

WRF 4.7.1 wrote ``wrfinput_initialized_d01`` after ``dfi_fst_init``'s ``start_domain``: its P/AL/P_HYD
are start_em's first-trip re-derivation from the filtered PH/MU/THM in the same file.  Two 6x6
column blocks (domain centre and the highest terrain) with all levels plus the base/metric inputs.

usage: python build_start_em_fixture.py <wrfinput_initialized_d01> <out.npz>
"""
import hashlib
import sys

import netCDF4
import numpy as np

COLUMN = ("PH", "MU", "THM", "P", "AL", "P_HYD", "PB", "PHB", "MUB", "HGT")
SCALAR = ("C3H", "C4H", "C3F", "C4F", "P_TOP", "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT")


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    with netCDF4.Dataset(src) as ds:
        hgt = np.asarray(ds.variables["HGT"][0])
        ny, nx = hgt.shape
        jm, im = np.unravel_index(np.argmax(hgt[3:-3, 3:-3]), (ny - 6, nx - 6))
        blocks = [(ny // 2 - 3, nx // 2 - 3), (int(jm), int(im))]
        payload = {"blocks": np.asarray(blocks, dtype=np.int32)}
        for name in COLUMN:
            arr = np.asarray(ds.variables[name][0])
            payload[name] = np.stack([arr[..., j0:j0 + 6, i0:i0 + 6] for j0, i0 in blocks])
        for name in SCALAR:
            payload[name] = np.asarray(ds.variables[name][0])
    payload["source_sha256"] = np.frombuffer(hashlib.sha256(open(src, "rb").read()).hexdigest().encode(), dtype=np.uint8)
    np.savez_compressed(out, **payload)
    print(out, {k: v.shape for k, v in payload.items()})


if __name__ == "__main__":
    main()
