"""w41-hailtables: mp38 DOUBLE tables as traced program operands, not baked HLO constants.

Light tests: released treedef, round trip, abstract lowering (no table load). The
BYTE gate loads the real ~0.97 GB tables and compiles the baked control (>6 GB RSS):
run it alone in a RAM slot (`--run-heavy`).
"""
from dataclasses import replace
from pathlib import Path
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.thompson_hail import column, tables


def _nl(option=38, **kwargs):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=option,
                               ra_sw_physics=0, ra_lw_physics=0, sf_sfclay_physics=0, bl_pbl_physics=0,
                               cu_physics=0, **kwargs)


def _tiny_tables(fill=1.):
    leaf = lambda: jnp.full((2,), fill, jnp.float64)
    return tables.ThompsonHailTables({n: leaf() for n in ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")},
                                     {n: leaf() for n in tables.COLD_OPERAND_NAMES}, leaf(), leaf())


def _column_spec(ncol=3, nz=8):
    from gpuwrf.physics.thompson_hail.column import ThompsonHailColumnState
    f = lambda: jax.ShapeDtypeStruct((ncol, nz), jnp.float32)
    names = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Nc", "nwfa", "nifa", "T", "p", "rho")
    return ThompsonHailColumnState(*(f() for _ in names), Ng=f(), dz=f(), w=f(), qvolg=f())


def _constant_bytes(text):
    """Bytes of dense f64/f32/i32 StableHLO constants in a lowered module."""
    total = 0
    for shape, dtype in re.findall(r"stablehlo\.constant dense<.*?> : tensor<([0-9x]*)x?(f64|f32|i32)>", text):
        size = int(np.prod([int(d) for d in shape.split("x") if d])) if shape else 1
        total += size * (8 if dtype == "f64" else 4)
    return total


def test_released_treedef_has_no_hail_child_or_aux():
    for option in (8, 38):
        leaves, tree = jax.tree.flatten(_nl(option))
        children, aux = _nl(option).tree_flatten()
        assert not any(isinstance(a, tuple) and a and a[0] == "thompson_hail_tables" for a in aux)
        explicit = jax.tree.flatten(_nl(option, thompson_hail_tables=None))
        assert explicit[1] == tree and len(explicit[0]) == len(leaves)


def test_bound_tables_round_trip_as_traced_children():
    base = _nl(38, aer_opt=0)
    on = replace(base, thompson_hail_tables=_tiny_tables(3.))
    leaves, tree = jax.tree.flatten(on)
    assert tree != jax.tree.flatten(base)[1]
    assert len(leaves) == len(jax.tree.leaves(base)) + 5 + len(tables.COLD_OPERAND_NAMES) + 2
    back = jax.tree.unflatten(tree, leaves)
    assert isinstance(back.thompson_hail_tables, tables.ThompsonHailTables)
    assert float(back.thompson_hail_tables.collision["tnr_gacr"][0]) == 3.
    # The hail child pops first; a trailing aer_opt child still round trips.
    static = jnp.arange(3.)
    with_aer = replace(on, aer_opt=1, tegen_aerosol=static)
    back = jax.tree.unflatten(*jax.tree.flatten(with_aer)[::-1])
    assert np.array_equal(back.tegen_aerosol, static) and back.thompson_hail_tables is not None


def test_bind_is_a_noop_for_other_options():
    nl = _nl(8)
    assert tables.bind_thompson_hail_tables(nl) is nl


def _assert_tables_are_operands():
    abstract = tables.abstract_tables()
    expected = sum(int(np.prod(leaf.shape)) * 8 for leaf in jax.tree.leaves(abstract))
    assert expected > 9.7e8
    fn = jax.jit(lambda state, t: column.step_thompson_hail_column_with_precip(state, 18., hail_tables=t))
    lowered = fn.lower(_column_spec(), abstract)
    text = lowered.as_text()
    main = next(line for line in text.splitlines() if "func.func public @main" in line)
    assert main.count("tensor<37x37x9x37x37xf64>") == 5, "collision tables must be entry parameters"
    assert main.count("tensor<3388275x4xf64>") == 1 and main.count("tensor<9157500x2xf64>") == 1
    assert _constant_bytes(text) < 1.0e7, _constant_bytes(text)


def test_operand_lowering_holds_no_table_constants():
    """Abstract operands: the step program carries the ~0.97 GB as parameters, not literals."""
    jax.clear_caches()
    _assert_tables_are_operands()


def test_lowering_gate_kills_a_baked_table_mutant(monkeypatch):
    """Mutant: HailContext ignores the operands and bakes (tiny patched) host tables."""
    from gpuwrf.physics.thompson_hail import context, native
    from gpuwrf.physics.thompson_tables import WrfColdCollectionTables
    tiny = lambda *shape: np.full(shape, .5, np.float64)
    monkeypatch.setattr(context, "collision_tables", lambda: {n: tiny(2, 2, 2, 2, 2) for n in
                        ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr")})
    monkeypatch.setattr(native, "_freezing_tables", lambda: {"qrfz4": tiny(2, 4), "qcfz": tiny(2, 2)})
    cold = WrfColdCollectionTables(**{n: tiny(2, 2) if n.endswith("_qcfz") else tiny(2, 2, 2, 2)
                                      for n in WrfColdCollectionTables._fields})
    original = column.HailContext
    def baked(state, tabs, _cold, *, raw_qv, operands=None):
        return original(state, tabs, cold, raw_qv=raw_qv)
    monkeypatch.setattr(column, "HailContext", baked)
    jax.clear_caches()
    try:
        with pytest.raises(AssertionError, match="entry parameters"):
            _assert_tables_are_operands()
    finally:
        jax.clear_caches()


def test_cheap_key_covers_table_bytes_while_hlo_takes_them_as_operands(monkeypatch):
    """t-aotkey rule: one changed table byte changes the full key; the HLO does not bake it."""
    from gpuwrf.physics.thompson_hail import context
    from gpuwrf.runtime import aot_cheap_key as ck, aot_executable as ax
    nl = replace(_nl(38), thompson_hail_tables=_tiny_tables(2.))
    x = jnp.asarray(1., jnp.float64)

    @jax.jit
    def fn(value, config):
        return value + config.thompson_hail_tables.collision["tcg_racg"][0]

    key = ck.cheap_key(fn, (x, nl), {}, nl)
    hlo = ax.hlo_sha256_from_lowered(fn.lower(x, nl))
    assert key is not None
    original = context.collision_tables()
    table = original["tcg_racg"].copy()
    table.flat[0] += table.dtype.type(1)
    changed = dict(original, tcg_racg=table)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(context, "collision_tables", lambda: changed)
            fn.clear_cache()
            assert tables.host_tables().collision["tcg_racg"] is table, "bind must read the keyed loader"
            assert ck.cheap_key(fn, (x, nl), {}, nl) != key, "a table byte did not change the full key"
            assert ax.hlo_sha256_from_lowered(fn.lower(x, nl)) == hlo, "table values leaked into the HLO"
    finally:
        fn.clear_cache()
    other = replace(nl, thompson_hail_tables=_tiny_tables(5.))
    assert ax.hlo_sha256_from_lowered(fn.lower(x, other)) == hlo
    assert float(fn(x, other)) == 6. and float(fn(x, nl)) == 3.


@pytest.mark.heavy
def test_operand_step_is_byte_equal_to_baked_on_original_columns():
    proof = Path(column.__file__).resolve().parents[4] / "proofs/v040/mpmisc/thompson38"
    sys.path.insert(0, str(proof))
    from check_columns import parse
    from gpuwrf.physics.thompson_hail.column import ThompsonHailColumnState
    refs = [parse(proof / f"case{c}.txt") for c in range(1, 7)]
    dt = refs[0]["scalars"]["DT"]
    cols = {name: np.asarray([r["columns"][name] for r in refs], np.float32) for name in refs[0]["columns"]}
    fields = {"qv": "QV", "qc": "QC", "qr": "QR", "qi": "QI", "qs": "QS", "qg": "QG", "Ni": "NI", "Nr": "NR",
              "Nc": "NC", "Ng": "NG", "qvolg": "QB", "nwfa": "NWFA", "nifa": "NIFA", "T": "T"}
    values = {field: jnp.asarray(cols[name + "_IN"]) for field, name in fields.items()}
    p, temp, qv = jnp.asarray(cols["P"]), values["T"], values["qv"]
    rho = np.float32(.622) * p / (np.float32(287.04) * temp * (qv + np.float32(.622)))
    w = jnp.asarray(cols["W_IN"]) if "W_IN" in cols else jnp.zeros_like(qv)
    state = ThompsonHailColumnState(**values, p=p, rho=rho, dz=jnp.asarray(cols["DELZ"]), w=w)
    baked = column.step_thompson_hail_column_with_precip(state, dt)
    bound = tables.bind_thompson_hail_tables(_nl(38)).thompson_hail_tables
    operand = column.step_thompson_hail_column_with_precip(state, dt, hail_tables=bound)
    for name, a, b in zip(("state", "precip"), baked, operand):
        for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree.leaves(b)):
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype and np.array_equal(x.view(np.uint8), y.view(np.uint8)), (name, path)
    # Deletion witness: one changed operand byte must reach the output.
    mutated = bound._replace(collision=dict(bound.collision, tcg_racg=bound.collision["tcg_racg"] * 2.))
    changed = column.step_thompson_hail_column_with_precip(state, dt, hail_tables=mutated)
    assert any(not np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree.leaves(operand), jax.tree.leaves(changed)))


# ---- checkpoints: the bound tables never enter a pickle payload ----------------------------------------

def _sealed_tiny(fill=1.):
    """What bind produces: claimed digest of the exact leaves, sealed to those leaf objects."""
    bound = _tiny_tables(fill)
    bound.digest = tables.content_digest(jax.tree.map(np.asarray, bound))
    return bound.seal()


def _assert_payload_holds_reference_only(hostify, grid):
    bound = _sealed_tiny()
    host = hostify(replace(_nl(38), thompson_hail_tables=bound), grid)
    assert isinstance(host.thompson_hail_tables, tables.ThompsonHailTablesRef), "tables reached the payload"
    assert host.thompson_hail_tables.digest == bound.digest


def _assert_changed_tables_refused(hostify, grid):
    """Critic FIX 00:39Z: a changed table must never checkpoint under its stale digest."""
    sealed = _sealed_tiny()
    replaced = sealed._replace(collision=dict(sealed.collision, tcg_racg=sealed.collision["tcg_racg"].at[0].set(0.)))
    mutated = _sealed_tiny()
    mutated.collision["tnr_gacr"] = mutated.collision["tnr_gacr"].at[0].set(2.)
    for case in (replaced, mutated):
        try:
            hostify(replace(_nl(38), thompson_hail_tables=case), grid)
        except ValueError as exc:
            assert "changed since bind" in str(exc)
            continue
        raise AssertionError("changed tables were checkpointed under a stale digest")
    # Unchanged contents in a rebuilt container are re-verified and accepted.
    same = sealed._replace(collision=dict(sealed.collision))
    assert hostify(replace(_nl(38), thompson_hail_tables=same), grid).thompson_hail_tables.digest == sealed.digest


@pytest.mark.parametrize("module", ["gpuwrf.runtime.checkpoint", "gpuwrf.io.restart"])
def test_pickle_helpers_store_a_reference_not_the_tables(module):
    import importlib
    helpers = importlib.import_module(module)
    _assert_payload_holds_reference_only(helpers._hostify_namelist, _nl(38).grid)
    # Released/other namelists keep the old payload exactly.
    plain = helpers._hostify_namelist(_nl(8), _nl(8).grid)
    assert plain.thompson_hail_tables is None


@pytest.mark.parametrize("module", ["gpuwrf.runtime.checkpoint", "gpuwrf.io.restart"])
def test_reference_gate_kills_a_hostify_everything_mutant(monkeypatch, module):
    import importlib
    helpers = importlib.import_module(module)
    monkeypatch.setattr(tables, "checkpoint_split", lambda namelist: (namelist, None))
    with pytest.raises(AssertionError, match="tables reached the payload"):
        _assert_payload_holds_reference_only(helpers._hostify_namelist, _nl(38).grid)


@pytest.mark.parametrize("module", ["gpuwrf.runtime.checkpoint", "gpuwrf.io.restart"])
def test_changed_tables_never_checkpoint_under_a_stale_digest(module):
    import importlib
    helpers = importlib.import_module(module)
    _assert_changed_tables_refused(helpers._hostify_namelist, _nl(38).grid)


@pytest.mark.parametrize("module", ["gpuwrf.runtime.checkpoint", "gpuwrf.io.restart"])
def test_stale_digest_gate_kills_a_trust_the_cache_mutant(monkeypatch, module):
    import importlib
    helpers = importlib.import_module(module)
    monkeypatch.setattr(tables.ThompsonHailTables, "verified_digest", lambda self: self.digest)
    with pytest.raises(AssertionError, match="stale digest"):
        _assert_changed_tables_refused(helpers._hostify_namelist, _nl(38).grid)


def test_resume_refuses_changed_loaders_before_any_device_transfer(monkeypatch):
    host = jax.tree.map(np.asarray, _tiny_tables(4.))
    monkeypatch.setattr(tables, "host_tables", lambda: host)
    def no_transfer(*args, **kwargs):
        raise RuntimeError("the ~1 GB tables were transferred before the digest check")
    monkeypatch.setattr(tables.jax, "device_put", no_transfer)
    with pytest.raises(ValueError, match="refusing to resume"):
        tables.checkpoint_restore(_nl(38), tables.ThompsonHailTablesRef("0" * 64))


def test_undigested_bound_tables_refuse_a_checkpoint():
    with pytest.raises(ValueError, match="content digest"):
        tables.checkpoint_split(replace(_nl(38), thompson_hail_tables=_tiny_tables()))


@pytest.mark.heavy
def test_mp38_checkpoint_is_small_and_resumes_byte_equal(tmp_path, monkeypatch):
    """Real tables: payload << 973 MB, resume re-provides identical bytes, step BYTE, refusal on changed loaders."""
    import pickle
    from test_ferrier_carry import grid
    from gpuwrf.contracts.state import State, Tendencies
    from gpuwrf.coupling.thompson_hail_adapter import thompson_hail_adapter
    from gpuwrf.physics.thompson_hail import context
    from gpuwrf.runtime.checkpoint import read_checkpoint, write_checkpoint
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.io import restart

    domain = grid()
    nl = OperationalNamelist(grid=domain, tendencies=Tendencies.zeros(domain), metrics=domain.metrics,
                             dt_s=18., acoustic_substeps=4, mp_physics=38)
    bound = tables.bind_thompson_hail_tables(nl)
    assert bound.thompson_hail_tables.digest == tables.content_digest(tables.host_tables())
    state = State.zeros(domain, mp_physics=38).ensure_conditional_leaves(mp_physics=38, use_mp_re=0)
    heights = jnp.broadcast_to(jnp.arange(9, dtype=jnp.float32)[:, None, None] * jnp.float32(250.), (9, 2, 2))
    state = state.replace(theta=jnp.full_like(state.theta, 268.), p=jnp.full_like(state.p, 70000.),
                          qv=jnp.full_like(state.qv, .004), qc=jnp.full_like(state.qc, 4e-4),
                          qr=jnp.full_like(state.qr, 3e-4), qg=jnp.full_like(state.qg, 2e-3),
                          Ng=jnp.full_like(state.Ng, 5e3), qvolg=jnp.full_like(state.qvolg, 4e-6),
                          Nr=jnp.full_like(state.Nr, 2e4), ph=heights * jnp.float32(9.81))
    path = write_checkpoint(state, bound, domain, 3, tmp_path / "mp38.ckpt")
    assert path.stat().st_size < 5_000_000, path.stat().st_size  # vs >= 973,433,592 B of arrays before
    with path.open("rb") as handle:
        assert isinstance(pickle.load(handle)["namelist"].thompson_hail_tables, tables.ThompsonHailTablesRef)
    _, restored, _, step = read_checkpoint(path)
    assert step == 3 and restored.thompson_hail_tables.digest == bound.thompson_hail_tables.digest
    for a, b in zip(jax.tree.leaves(bound.thompson_hail_tables), jax.tree.leaves(restored.thompson_hail_tables)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    # io.restart shares the same reference contract.
    back = restart._restore_namelist(restart._hostify_namelist(bound, domain), domain)
    assert back.thompson_hail_tables.digest == bound.thompson_hail_tables.digest
    # Resumed step with the re-provided tables is bit-identical to the uninterrupted one.
    run = lambda n: thompson_hail_adapter(state, 18., domain, return_precipitation=True, hail_tables=n.thompson_hail_tables)
    for x, y in zip(jax.tree.leaves(run(bound)), jax.tree.leaves(run(restored))):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    # A bound table changed after bind keeps its claimed digest only until a checkpoint re-verifies it.
    stale = bound.thompson_hail_tables._replace(collision=dict(bound.thompson_hail_tables.collision,
        tcg_racg=bound.thompson_hail_tables.collision["tcg_racg"].at[0, 0, 0, 0, 0].add(1.)))
    with pytest.raises(ValueError, match="changed since bind"):
        write_checkpoint(state, replace(bound, thompson_hail_tables=stale), domain, 3, tmp_path / "stale.ckpt")
    # Changed loader bytes since the checkpoint: refuse to resume.
    original = context.collision_tables()
    changed = original["tnr_gacr"].copy()
    changed.flat[7] += changed.dtype.type(1)
    monkeypatch.setattr(context, "collision_tables", lambda: dict(original, tnr_gacr=changed))
    with pytest.raises(ValueError, match="refusing to resume"):
        read_checkpoint(path)
