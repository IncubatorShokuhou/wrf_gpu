"""mp38 DOUBLE lookup tables as traced program operands instead of baked constants.

Original module_mp_thompson.F (:375-385) holds every table below as REAL(KIND=8).
Read inside a trace, the cached host loaders become ~0.96 GB of f64 HLO constants
(t-lowersweep c65). Bound once per run as an OperationalNamelist traced child,
they are device-resident parameters of the step program; the values are the
SAME host loaders that aot_baked_tables keys for mp_physics=38, so the cheap key
still changes with any table byte. Absent (None), HailContext keeps the baked
path unchanged.

Checkpoints never carry the ~0.97 GB: the pickle restart helpers store a
ThompsonHailTablesRef (content digest) and re-provide the tables from the keyed
loaders on resume, refusing if the loader bytes no longer match.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax


# mp8-shared cold reads that mp38 does not override: qr_acr_qsV2 rain-snow and
# the freezeH2O qcfz planes. The five graupel names are the hail collision set.
COLD_OPERAND_NAMES = (
    "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
    "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
    "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
    "tpi_qcfz", "tni_qcfz",
)
COLLISION_NAMES = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")


@jax.tree_util.register_pytree_node_class
class ThompsonHailTables:
    """collision (37, 37, 9, 37, 37) x5, cold COLD_OPERAND_NAMES, qrfz4 (-1, 4), qcfz (-1, 2).

    ``digest`` is static aux (content of the host arrays bound), so it rides the
    treedef and the checkpoint reference, never the traced program. It is only a
    CLAIM: bind seals the exact (immutable) leaf objects it was computed for, and
    verified_digest() recomputes it from the actual leaves whenever they are not
    those sealed ones (_replace, mutated dict children, rebuilt containers).
    """

    __slots__ = ("collision", "cold", "qrfz4", "qcfz", "digest", "_sealed")

    def __init__(self, collision, cold, qrfz4, qcfz, digest=None):
        self.collision, self.cold, self.qrfz4, self.qcfz, self.digest = collision, cold, qrfz4, qcfz, digest
        self._sealed = None

    def tree_flatten(self):
        return (self.collision, self.cold, self.qrfz4, self.qcfz), self.digest

    @classmethod
    def tree_unflatten(cls, digest, children):
        return cls(*children, digest=digest)

    def _replace(self, **changes):
        values = {name: getattr(self, name) for name in ("collision", "cold", "qrfz4", "qcfz", "digest")}
        values.update(changes)
        return type(self)(**values)  # never sealed: a replaced container is re-verified

    def seal(self):
        """Pin the claimed digest to these exact leaf objects (bind only; host work, never per step)."""
        self._sealed = tuple(jax.tree.leaves(self))
        return self

    def verified_digest(self) -> str:
        """The digest of the CURRENT leaves: cached if they are the sealed objects, else recomputed (host copy)."""
        import numpy as np

        leaves = jax.tree.leaves(self)
        if (self.digest and self._sealed is not None and len(leaves) == len(self._sealed)
                and all(a is b for a, b in zip(leaves, self._sealed))):
            return self.digest
        return content_digest(jax.tree.map(np.asarray, self))


@dataclass(frozen=True)
class ThompsonHailTablesRef:
    """Checkpoint stand-in for bound tables: content digest only (no arrays)."""
    digest: str


def content_digest(tables: ThompsonHailTables) -> str:
    """SHA-256 over every host array (name, shape, dtype, bytes) in a fixed order."""
    import hashlib

    import numpy as np

    h = hashlib.sha256(b"thompson-hail-tables-v1")
    named = [("collision/" + n, tables.collision[n]) for n in COLLISION_NAMES]
    named += [("cold/" + n, tables.cold[n]) for n in COLD_OPERAND_NAMES]
    named += [("qrfz4", tables.qrfz4), ("qcfz", tables.qcfz)]
    for name, value in named:
        value = np.ascontiguousarray(value)
        h.update(f"{name}|{value.shape}|{value.dtype.str}|".encode())
        h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def host_tables() -> ThompsonHailTables:
    """Read-only NumPy f64 values from the cached loaders that the AOT key hashes."""
    from gpuwrf.physics.thompson_tables import load_wrf_cold_collection_tables
    from .context import collision_tables
    from .native import _freezing_tables

    cold = load_wrf_cold_collection_tables()
    freezing = _freezing_tables()
    collision = collision_tables()
    return ThompsonHailTables({name: collision[name] for name in COLLISION_NAMES},
                              {name: getattr(cold, name) for name in COLD_OPERAND_NAMES},
                              freezing["qrfz4"], freezing["qcfz"])


def bind_thompson_hail_tables(namelist, device=None, *, expected_digest=None):
    """Attach the device-resident mp38 tables to ``namelist`` (other options: unchanged)."""
    from dataclasses import replace

    import jax.numpy as jnp
    import numpy as np

    if int(namelist.mp_physics) != 38:
        return namelist
    if jax.dtypes.canonicalize_dtype(jnp.float64) != jnp.float64:
        raise ValueError("mp_physics=38 needs jax_enable_x64: the original Thompson tables are DOUBLE")
    if device is None:
        from gpuwrf.runtime.operational_mode import _operational_device
        device = _operational_device()
    host = host_tables()
    if any(np.asarray(value).dtype != np.float64 for value in jax.tree.leaves(host)):
        raise ValueError("mp_physics=38 tables must be the original DOUBLE values")
    host.digest = content_digest(host)
    if expected_digest is not None and host.digest != expected_digest:
        # Checked BEFORE the ~1 GB device transfer.
        raise ValueError(f"mp38 checkpoint was written with Thompson tables {expected_digest[:16]}..., the current "
                         f"loaders provide {host.digest[:16]}...; refusing to resume with different tables")
    # Committed once here: per-call dispatch then aliases, never re-transfers.
    return replace(namelist, thompson_hail_tables=jax.device_put(host, device).seal())


def checkpoint_split(namelist):
    """(namelist without bound tables, ThompsonHailTablesRef | None) for a pickle payload."""
    from dataclasses import is_dataclass, replace

    tables = getattr(namelist, "thompson_hail_tables", None) if is_dataclass(namelist) else None
    if tables is None:
        return namelist, None
    if isinstance(tables, ThompsonHailTablesRef):  # a loaded payload: strip before any tree_map
        return replace(namelist, thompson_hail_tables=None), tables
    if not isinstance(tables, ThompsonHailTables) or not tables.digest:
        raise ValueError("mp38 checkpoint: bound Thompson tables carry no content digest (bind them via "
                         "bind_thompson_hail_tables)")
    actual = tables.verified_digest()
    if actual != tables.digest:
        raise ValueError("mp38 checkpoint: the bound Thompson tables changed since bind (content digest "
                         f"{actual[:16]}... != recorded {tables.digest[:16]}...); refusing to checkpoint a stale identity")
    return replace(namelist, thompson_hail_tables=None), ThompsonHailTablesRef(actual)


def checkpoint_restore(namelist, ref):
    """Re-provide the tables from the keyed loaders; REFUSE if their bytes changed since the checkpoint."""
    if ref is None:
        return namelist
    if int(namelist.mp_physics) != 38:
        raise ValueError("mp38 table checkpoint reference on a non-mp38 namelist; refusing to resume")
    return bind_thompson_hail_tables(namelist, expected_digest=ref.digest)


def abstract_tables() -> ThompsonHailTables:
    """ShapeDtypeStruct specs (lowering preflight without loading ~0.97 GB)."""
    import numpy as np

    spec = lambda *shape: jax.ShapeDtypeStruct(shape, np.float64)
    collision = {name: spec(37, 37, 9, 37, 37) for name in COLLISION_NAMES}
    cold = {name: spec(37, 45) if name.endswith("_qcfz") else spec(37, 9, 37, 37) for name in COLD_OPERAND_NAMES}
    return ThompsonHailTables(collision, cold, spec(37 * 37 * 45 * 55, 4), spec(37 * 100 * 45 * 55, 2))
