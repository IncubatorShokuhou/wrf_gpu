"""Independent history replay with WRF phy_prep pressure/dz/density operands.

Reuses the frozen public-driver executable and selected real columns, but reads
their original history frames and applies the literal WRF pressure integration.
History does not publish held WSPD, so that worker approximation remains explicit.
This is a caller-operand sensitivity check, not an exact live savepoint claim.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

root, oracle, output = map(lambda p: Path(p).resolve(), sys.argv[1:4])
sys.path[:0] = [str(root / "src"), str(root / "tests/v025/t_cfix_mynn")]
import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
import f20_oracle as O
import f20_port as F
from f20_public_kernel import load_sets

assert jax.devices()[0].platform == "cpu"
spec = importlib.util.spec_from_file_location("f20_review_helpers", root / "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
f = np.float32
result = {"scope": "original-history replay; hydrostatic p/psfc, literal dz8w, hypsometric ALT rho; held WSPD approximation retained",
          "source_commit": "b8fab854ff9d7d6f42d41a959c16529aa227422d", "sets": {}}
for name, (columns, meta) in load_sets().items():
    cols = {key: val.copy() for key, val in columns.items()}
    nz, ny, nx = meta["shape"]
    index = cols["columns"]
    def pick(a):
        return np.ascontiguousarray(a.reshape(a.shape[0], ny * nx)[:, index].T)
    with Dataset(meta["source"]) as nc:
        def v(key):
            return np.asarray(nc.variables[key][0], np.float32)
        assert nc.HYPSOMETRIC_OPT == 2
        assert np.array_equal(pick(v("W")), cols["w"])
        assert np.array_equal(pick(v("T") + f(300)), cols["th"])
        assert np.array_equal(v("QFX").ravel()[index], cols["qfx"])
        assert np.array_equal(v("HFX").ravel()[index], cols["hfx"])
        mut = (v("MU") + v("MUB")).ravel()[index]
        c1h, c2h, dnw, ptop = v("C1H"), v("C2H"), v("DNW"), v("P_TOP")
        qt = np.zeros((len(index), nz), np.float32)
        for key in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            qt = qt + pick(v(key))
        faces = np.empty((len(index), nz + 1), np.float32)
        faces[:, -1] = ptop
        for k in range(nz - 1, -1, -1):
            faces[:, k] = faces[:, k + 1] - (f(1) + qt[:, k]) * (c1h[k] * mut + c2h[k]) * dnw[k]
        cols["p"] = f(.5) * (faces[:, :-1] + faces[:, 1:])
        cols["psfc"] = faces[:, 0]
        phi = pick(v("PH") + v("PHB"))
        zface = phi / f(9.81)
        cols["dz"] = zface[:, 1:] - zface[:, :-1]
        c3f, c4f, c3h, c4h = v("C3F"), v("C4F"), v("C3H"), v("C4H")
        p_up = c3f[1:] * mut[:, None] + c4f[1:] + ptop
        p_down = c3f[:-1] * mut[:, None] + c4f[:-1] + ptop
        p_mid = c3h * mut[:, None] + c4h + ptop
        alt = (phi[:, 1:] - phi[:, :-1]) / p_mid / np.log(p_down / p_up)
        cols["rho"] = (f(1) + cols["qv"]) / alt
    for value in cols.values():
        assert np.isfinite(value).all()
    work = output.parent / "caller_operands_work" / name
    wrf, log = O.run(oracle, work, cols, meta["dt"], meta["dx"])
    psig, expected = T._port_psig(cols, log, meta["dx"], cols["w"][:, :nz])
    assert np.array_equal(T._bits(psig), T._bits(expected))
    port = F.run_kernel(cols, meta["dt"], meta["dx"])
    difference = np.abs(port["maxmf"] - wrf["maxmf"])
    assert np.isfinite(difference).all()
    bound = f(2e-4) + f(1e-3) * np.abs(wrf["maxmf"])
    row = dict(n=len(index), taper_byte=True, maxmf_max_abs=float(difference.max()),
               maxmf_gate=bool(np.all(difference <= bound)),
               max_pressure_operand_change_pa=float(np.abs(cols["p"] - columns["p"]).max()),
               max_density_operand_change=float(np.abs(cols["rho"] - columns["rho"]).max()),
               max_dz_operand_change_m=float(np.abs(cols["dz"] - columns["dz"]).max()))
    result["sets"][name] = row
    print(name, row, flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
assert all(row["maxmf_gate"] for row in result["sets"].values())
