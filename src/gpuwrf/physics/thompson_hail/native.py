"""Private native-REAL bindings; existing Thompson code objects are reused."""
from types import FunctionType, SimpleNamespace
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from gpuwrf.assets import asset_path, ensure_asset
from gpuwrf.physics import thompson_column as original
from gpuwrf.physics import thompson_aero_column as original_aero
from gpuwrf.physics.thompson_aero_tables import THOMPSON_AERO_TABLES


def namespace(module, replacements):
    env = dict(vars(module))
    env.update(replacements)
    for name, function in tuple(env.items()):
        if isinstance(function, FunctionType) and function.__module__ == module.__name__:
            copied = FunctionType(function.__code__, env, function.__name__, function.__defaults__, function.__closure__)
            copied.__kwdefaults__ = function.__kwdefaults__
            env[name] = copied
    return SimpleNamespace(**env)


TC = namespace(original, {
    "_native_real_enabled": lambda: True,
    "_wrf_constants_enabled": lambda: True,
    "_mixed_phase_wrf_enabled": lambda: True,
    "_ice_collection_enabled": lambda: True,
    "_riming_enabled": lambda: True,
    "_cold_collection_enabled": lambda: True,
    "_sed_prep_fused_enabled": lambda: False,
    "_implicit_sed_nsub": lambda: 0,
    "_full_column_enabled": lambda: False,
    "_column_sed_enabled": lambda: False,
    # Original REAL :812 must round before promotion into DOUBLE ilamg
    # terms. Casting the old Python .86 only after multiplication is late.
    "T1_SUBL_QG": np.float32(.8600000143051147),
    # Original REAL cie(2) division forms the edge slope before its
    # assignment to DOUBLE lami (:1855-1871). A Python REAL8 host fold
    # changes the reconstructed working ice number (H38-49 savepoint).
    "CIE2": np.float32(4.),
})

@lru_cache(maxsize=1)
def _cloud_coefficients():
    # Original initialized REAL WGAMMA coefficients, independently dumped
    # by GDB. The legacy Aero asset's REAL8 values cannot substitute them.
    with np.load(Path(__file__).with_name("cloud_coefficients.npz"),allow_pickle=False) as data:
        result={name:np.asarray(data[name],np.float32) for name in ("cce","ccg","ocg1","ocg2")}
    for value in result.values():
        value.flags.writeable=False
    return result


@lru_cache(maxsize=1)
def _freezing_tables():
    # Source4630-4635 reads these same six DOUBLE records. Keep the mp28
    # legacy tables unchanged; private38 consumes only the original values.
    arrays = {}
    for field, shape, names in (
        ("qrfz4", (37, 37, 45, 55, 4), ("tpi_qrfz", "tpg_qrfz", "tni_qrfz", "tnr_qrfz")),
        ("qcfz", (37, 100, 45, 55, 2), ("tpi_qcfz", "tni_qcfz")),
    ):
        values = np.empty(shape, np.float64)
        for channel, name in enumerate(names):
            asset = ensure_asset(asset_path("src/gpuwrf/physics/thompson_hail/freezing_tables_" + name + ".npz"))
            with np.load(asset, allow_pickle=False) as packed:
                piece = packed["values"]
                if piece.dtype != np.float64 or piece.shape != shape[:-1]:
                    raise ValueError("invalid original Thompson38 freezing table " + name)
                values[..., channel] = piece
                del piece
        values.flags.writeable = False
        arrays[field] = values.reshape((-1, shape[-1]))
    return arrays


def real_tables(operands=None):
    # Called per trace, never cache a cast tracer in a module global (E206).
    # ``operands`` (tables.ThompsonHailTables) supplies the DOUBLE freezing
    # tables as traced program operands instead of baked constants.
    tables = THOMPSON_AERO_TABLES._replace(**{
        name: getattr(THOMPSON_AERO_TABLES, name).astype(np.float32)
        for name in ("ccn_act", "ta_na", "ta_ww", "cce", "ccg", "ocg1", "ocg2")
    }, nic1=THOMPSON_AERO_TABLES.nic1.astype(np.int32))
    if operands is not None:
        return tables._replace(**{name:jnp.asarray(value) for name,value in _cloud_coefficients().items()},
                               qrfz4=operands.qrfz4, qcfz=operands.qcfz)
    return tables._replace(**{name:jnp.asarray(value) for name,value in
        {**_cloud_coefficients(), **_freezing_tables()}.items()})

replacements = {name: getattr(TC, name) for name, function in vars(original_aero).items()
                if isinstance(function, FunctionType) and function.__module__ == original.__name__ and hasattr(TC, name)}
replacements.update({"THOMPSON_AERO_TABLES": THOMPSON_AERO_TABLES,
                     "AM_R": np.float32(original.WRF_REAL_CONSTANTS["AM_R"]),
                     "AM_I": np.float32(original.WRF_REAL_CONSTANTS["AM_I"]),
                     # Original :213, read-only GDB H38-32 parameter witness.
                     "AR_VOLUME": np.float32(6.544983878764576e-17),
                     "R_UNI": np.float32(8.314),
                     # Original iceDeMott REAL parameter (:5472), confirmed
                     # by original GDB H38-42. The REAL8 host fold differs.
                     "RHO_NOT0": np.float32(1.2922837734222412),
                     # Initialized pristine REAL scalars, read-only GDB at
                     # :1648 after init: bits3f47ae14 /42135207 (H38-38).
                     "T1_QR_EV": np.float32(.7799999713897705),
                     "T2_QR_EV": np.float32(36.83010482788086),
                     "OBMR": np.float32(1.) / np.float32(3.),
                     "OBMI": np.float32(1.) / np.float32(3.)})
AERO = namespace(original_aero, replacements)
