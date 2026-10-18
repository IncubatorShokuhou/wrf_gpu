"""AE01 fixtures: pristine WRF V4.7.1 aer_opt=1 oracles (proofs/w41_aerosol, README there).

``tegen_units_v1.npz``: swaerpr tables after rrtmg_swinit; aerosol_in on 51 x 53 lat/lon points incl. table points and
both periodic edges (full array pinned by ``aerodm_sha256``); aer_time_int factors for 520 julians (unit-vector aerodm:
row = fact1 at nm + fact2 at np); aer_p_int on 31 real WN3 0227 + 6 synthetic columns (above/below the table, exact
table-level hits).  ``tegen_sw_0227_v1.npz``: RRTMG_SWRAD with the RE01 arm-A caller wiring (Thompson radii,
cldovrlp 2) on the 186 RE01 columns for 11 arms (julian, aer_opt, aerod scale).  ``thompson_aer_sw_0227_v1.npz``
(AE07): aer_opt=3 -- gt_aod / calc_aerosol_rrtmg_sw optics (taod5503d, taod5502d, tauaer/ssaaer/asyaer) and
RRTMG_SWRAD on the same columns with WRF-default QNWFA/QNIFA profiles (in_qnwfa/in_qnifa) for 6 arms.
``thompson_aer_sw_0227_ens_v1.npz`` (AE09): the same pristine chain on a 32-member 1-ulp ensemble (T, QNWFA) of
every column and arm; only the cells whose WRF self-spread leaves the frozen bound are kept (index, ensemble min/max).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("fixtures")
UNITS = HERE / "tegen_units_v1.npz"
SW = HERE / "tegen_sw_0227_v1.npz"
AER3 = HERE / "thompson_aer_sw_0227_v1.npz"  # AE07 (aer_opt=3, proofs/w41_aerosol/ae07_*)
AER3_ENS = HERE / "thompson_aer_sw_0227_ens_v1.npz"  # AE09 (proofs/w41_aerosol/ae09_ensemble.py)
SHA = {UNITS.name: "0c52ab179fcd9f27cb17253d9ea19f1da91bb204bcac5f14d66d8179d6cf2159",
       SW.name: "4505d5cf41d1b3096cee3bb5d6d1504731470dcb4aae3b82735ae74db4db0047",
       "thompson_aer_sw_0227_v1.npz": "c3cd5c0732f0b31f2ba8ddc47066e0d0972eeb46e648fcd59b0345569e87b750",  # AE07
       "thompson_aer_sw_0227_ens_v1.npz": "db5e114f7041bf24ff03cb973c5e49f0dbc28449490b9f25848329aa8d2b3580",  # AE09
       "rrtmg_mp_re_0227_v1.npz": "bba1a3e080010dbadbe39c48d07113858220d104faf18bb72f0735a5ee541ac0"}  # RE01 operands
RE01 = Path(__file__).resolve().parents[1] / "fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
FLUX_BOUND = (1.0, 0.05)      # frozen tier-1 RRTMG bounds (fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml)
HEATING_BOUND = (1.0e-4, 0.05)


def load(path: Path) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SHA[Path(path).name]:
        raise ValueError(f"{path}: sha256 {digest} is not the registered AE01 fixture")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def bound_ratio(reference, candidate, bound) -> np.ndarray:
    """|candidate - reference| / (abs + rel |reference|); NaN/Inf candidates or references are outside (E200)."""

    ref = np.asarray(reference, np.float64)
    cand = np.asarray(candidate, np.float64)
    ratio = np.abs(cand - ref) / (bound[0] + bound[1] * np.abs(ref))
    return np.where(np.isfinite(ratio), ratio, np.inf)
