"""W2 admission: real-column consumer activity, class axes and coupled bounds."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proofs/w44_lsmparams"))
from columns import build
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.surface_params import controls as sc


@pytest.fixture(scope="module")
def nl():
    return build(options=NoahMPOptions(opt_crs=2))[5]


@pytest.mark.parametrize("name", sorted(sc.LSM_CONTROLS))
def test_prior_is_bound_consumer_and_rejects_nonfinite(nl, name, research_w2_policy):
    value = sc.prior_controls(nl, [name], device=False)[name]
    assert value.shape == sc.expected_shape(name, nl)
    assert value.tobytes() == np.asarray(sc._target(name, nl)).tobytes()
    for nonfinite in (np.nan, np.inf, -np.inf):
        bad = value.copy()
        bad.flat[0] = nonfinite
        with pytest.raises(ValueError, match="non-finite"):
            sc.validate_controls({name: bad}, nl)


def test_rsmin_requires_jarvis_and_soil_requires_table_schaake(nl, research_w2_policy):
    from types import SimpleNamespace
    fields = vars(nl)
    for opts, name, error in [(None, "rsmin", "Jarvis"),
                              (NoahMPOptions(opt_run=5), "dksat", "opt_run=3"),
                              (NoahMPOptions(opt_soil=2), "bexp", "opt_soil=1")]:
        inactive = SimpleNamespace(**{**fields, "noahmp_options": opts})
        with pytest.raises(ValueError, match=error):
            sc.expand_controls(inactive, {name: 1.})


@pytest.mark.parametrize("option,values", [("opt_irr", (1, 2, 3)), ("opt_tdrn", (1, 2))])
@pytest.mark.parametrize("name", sorted(sc.LSM_CONTROLS.keys() & {"smcmax", "dksat", "bexp", "psisat"}))
def test_soil_controls_refuse_unwired_optional_parameter_copies(nl, option, values, name, research_w2_policy):
    from dataclasses import replace
    from types import SimpleNamespace
    for value in values:
        extra = {"drain_layer_opt": 0} if option == "opt_tdrn" else {}  # WRF requires a DRAIN_LAYER_OPT
        options = replace(nl.noahmp_options, **{option: value}, **extra)
        unsupported = SimpleNamespace(**{**vars(nl), "noahmp_options": options})
        with pytest.raises(ValueError, match=option + "=0"):
            sc.expand_controls(unsupported, {name: 1.1}, device=False)


@pytest.mark.parametrize("name,factor", [("smcmax", 1.02), ("bexp", .8), ("psisat", 1.1)])
def test_changed_frozen_init_partition_refused_but_identity_and_retained_allowed(nl, name, factor, research_w2_policy):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    temperature = np.asarray(nl.noahmp_land.tslb).copy()
    temperature[0, 0, 0] = 270.35  # retained WRF frozen_grass10 temperature
    land = nl.noahmp_land.replace(tslb=temperature)
    frozen = OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True,
        noahmp_static=nl.noahmp_static, noahmp_energy_params=nl.noahmp_energy_params,
        noahmp_rad_params=nl.noahmp_rad_params, noahmp_options=nl.noahmp_options, noahmp_land=land)
    from gpuwrf.surface_params.lsmparams import active_columns
    assert active_columns(frozen).flat[0], "the frozen layer must sit on an active land column"
    changed = sc.expand_controls(frozen, {name: factor}, device=False)
    with pytest.raises(ValueError, match="NOAHMP_INIT SH2O"):
        sc.bind_domain_controls(frozen, {"d01": changed}, "d01")
    prior = sc.prior_controls(frozen, [name], device=False)
    sc.bind_domain_controls(frozen, {"d01": prior}, "d01")
    # Restored SH2O belongs to the checkpoint; parameter branches are checked
    # against retained porosity, not reinterpreted as a fresh NOAHMP_INIT.
    sc.bind_domain_controls(frozen, {"d01": changed}, "d01", cold_start=False)


def test_hvb_must_not_exceed_hvt(nl, research_w2_policy):
    fields = sc.prior_controls(nl, ["hvt", "hvb"], device=False)
    fields["hvb"][0, 0] = fields["hvt"][0, 0] + 1
    with pytest.raises(ValueError, match="HVB <= HVT"):
        sc.validate_controls(fields, nl)


def test_soil_class_mapping_uses_isltyp_not_vegetation(nl, research_w2_policy):
    # All four real inputs have soil class6, but different vegetation classes.
    control = sc.expand_controls(nl, {"dksat": {6: 2., 5: 9.}}, device=False)["dksat"]
    expected = (sc.table_base("dksat", nl) * 2).astype(control.dtype)
    np.testing.assert_array_equal(control, expected)
    for invalid in (0, 20, -1, 1.5):
        with pytest.raises(ValueError, match="1-based"):
            sc.expand_controls(nl, {"dksat": {invalid: 2.}}, device=False)


def test_porosity_edit_rechecks_existing_moisture_via_shared_validator(nl, monkeypatch, research_w2_policy):
    from gpuwrf.io import wrfinput_consistency as shared
    calls = []
    actual = shared.validate_soil_porosity

    def observe(moisture, cap, landmask, veg, **kwargs):
        calls.append(np.asarray(cap).copy())
        return actual(moisture, cap, landmask, veg, **kwargs)

    monkeypatch.setattr(shared, "validate_soil_porosity", observe)
    prior = sc.prior_controls(nl, ["smcmax"], device=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], prior["smcmax"])
    cap = prior["smcmax"].copy()
    # Preserve SMCREF ordering while putting existing water above the new cap.
    from types import SimpleNamespace
    wet_land = nl.noahmp_land.replace(smois=cap + np.asarray(.01, cap.dtype))
    wet = SimpleNamespace(**{**vars(nl), "noahmp_land": wet_land})
    with pytest.raises(ValueError, match="soil_porosity"):
        sc.validate_controls(prior, wet)
    assert len(calls) == 2


def test_legacy_prior_set_does_not_silently_enable_w2(nl):
    prior = sc.prior_controls(build()[5], device=False)
    assert set(prior) == set(sc.CONTROLS)
    assert not set(prior) & set(sc.LSM_CONTROLS)


@pytest.mark.parametrize("category", ["table", 51, 61])
def test_all_wrf_urban_classes_preserve_fixed_porosity(nl, category, tmp_path, monkeypatch, research_w2_policy):
    from types import SimpleNamespace
    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
    from gpuwrf.physics.noahmp.precision import real_tree
    from gpuwrf.surface_params.cli import write_basis, expand_from_basis
    monkeypatch.setenv("GPUWRF_NOAH_URBAN_SOIL_PARAMS", "1")
    vegetation = np.asarray(nl.noahmp_static.ivgtyp).copy()
    vegetation[0, 0] = int(nl.noahmp_static.parameters.isurban) if category == "table" else category
    static = nl.noahmp_static.replace(ivgtyp=vegetation)
    energy, radiation = real_tree(build_energy_params(static, nl.noahmp_land.tv.shape))
    urban = SimpleNamespace(**{**vars(nl), "noahmp_static": static,
                               "noahmp_energy_params": energy, "noahmp_rad_params": radiation})
    cap = sc.expand_controls(urban, {"smcmax": 1.4}, device=False)["smcmax"]
    np.testing.assert_array_equal(cap[:, 0, 0], np.full(4, .45, cap.dtype))
    wrong = cap.copy()
    wrong[:, 0, 0] = .46
    with pytest.raises(ValueError, match="urban porosity"):
        sc.validate_controls({"smcmax": wrong}, urban)
    path = tmp_path / "basis.npz"
    manifest = write_basis({"d01": urban}, path)
    with np.load(path, allow_pickle=False) as basis:
        expanded = expand_from_basis(basis, manifest, {"d01": {"smcmax": 1.4}})
    np.testing.assert_array_equal(expanded["d01"]["smcmax"], cap)


def test_glacier_partition_does_not_treat_month_table_as_domain(nl):
    from gpuwrf.coupling.glacier_surface_hook import _column_parameters
    p = nl.noahmp_static.parameters
    # This exact shape matches MODIS LAIM/SAIM (21 categories x 12 months).
    assert p.laim.shape == (21, 12)
    assert _column_parameters(p, np.array([0, 8]), (21, 12)) is p


def test_glacier_partition_subsets_every_w2_field_only(nl, research_w2_policy):
    from gpuwrf.coupling.glacier_surface_hook import _column_parameters
    values = sc.prior_controls(nl, ["smcmax", "bexp", "dksat", "psisat", "hvt", "hvb", "rsmin"])
    s, _, _ = sc.apply_surface_controls(nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params, values)
    indices = np.array([0, 2])
    subset = _column_parameters(s.parameters, indices, (1, 4))
    for name in values:
        np.testing.assert_array_equal(np.asarray(getattr(subset, name)), np.asarray(values[name])[..., indices])
    assert subset.laim is s.parameters.laim
    assert subset.saim is s.parameters.saim


def test_cli_basis_lists_w2_and_preserves_soil_class_axis(nl, tmp_path, research_w2_policy):
    from gpuwrf.surface_params import cli
    path = tmp_path / "basis.npz"
    manifest = cli.write_basis({"d01": nl}, path)
    assert set(sc.LSM_CONTROLS) <= set(manifest["domains"]["d01"]["controls"])
    with np.load(path, allow_pickle=False) as basis:
        factors = {"d01": {"dksat": {6: 1.5}}}
        got = cli.expand_from_basis(basis, manifest, factors)["d01"]
        want = sc.expand_controls(nl, factors["d01"], device=False)
        assert got["dksat"].tobytes() == want["dksat"].tobytes()
        with pytest.raises(ValueError, match="1-based"):
            cli.expand_from_basis(basis, manifest, {"d01": {"dksat": {0: 2.}}})


def test_retained_branch_soil_is_checked_without_mutating_original_holder(nl, research_w2_policy):
    from types import SimpleNamespace
    from gpuwrf.surface_params.lsmparams import validate_retained_porosity
    fields = {"smois": np.asarray(nl.noahmp_land.smois), "sh2o": np.asarray(nl.noahmp_land.sh2o)}
    controls = sc.prior_controls(nl, ["smcmax"], device=False)
    requested = SimpleNamespace(**{**vars(nl), "surface_controls": controls})
    original_land, original_static = requested.noahmp_land, requested.noahmp_static
    healthy = validate_retained_porosity(requested, fields)
    assert healthy["passed"] and healthy["check"] == "soil_porosity"
    wet = {**fields, "smois": controls["smcmax"] + np.asarray(.01, controls["smcmax"].dtype)}
    with pytest.raises(ValueError, match="soil_porosity"):
        validate_retained_porosity(requested, wet)
    assert requested.noahmp_land is original_land and requested.noahmp_static is original_static
    assert np.array_equal(fields["smois"], np.asarray(original_land.smois))
    with pytest.raises(ValueError, match="retained SMOIS and SH2O"):
        validate_retained_porosity(requested, None)
    assert validate_retained_porosity(nl, None) is None


def test_water_placeholders_are_preserved_without_allowing_zero_on_land(nl, tmp_path, research_w2_policy):
    from types import SimpleNamespace
    import jax.numpy as jnp
    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
    from gpuwrf.physics.noahmp.precision import real_tree
    from gpuwrf.surface_params import cli
    s = nl.noahmp_static
    def cell(field, value):
        array = np.array(getattr(s, field))
        array[0, 0] = value
        return jnp.asarray(array)
    water = s.replace(isltyp=cell("isltyp", 14), ivgtyp=cell("ivgtyp", 17),
                      landmask=cell("landmask", 0), xland=cell("xland", 2))
    ep, rp = real_tree(build_energy_params(water, (1, 4)))
    coastal = SimpleNamespace(**{**vars(nl), "noahmp_static": water, "noahmp_energy_params": ep,
                                 "noahmp_rad_params": rp})
    names = ("smcmax", "dksat", "bexp", "psisat")
    prior = sc.prior_controls(coastal, names, device=False)
    changed = sc.expand_controls(coastal, {name: 1.1 for name in names}, device=False)
    for name in names:
        assert changed[name][:, 0, 0].tobytes() == prior[name][:, 0, 0].tobytes()
    assert prior["bexp"][0, 0, 0] == prior["dksat"][0, 0, 0] == prior["psisat"][0, 0, 0] == 0.
    wrong = {"bexp": prior["bexp"].copy()}
    wrong["bexp"][:, 0, 0] = 1.
    with pytest.raises(ValueError, match="inactive"):
        sc.validate_controls(wrong, coastal)
    wrong["bexp"] = prior["bexp"].copy()
    wrong["bexp"][:, 0, 1] = 0.
    with pytest.raises(ValueError, match="outside"):
        sc.validate_controls(wrong, coastal)
    path = tmp_path / "coastal_basis.npz"
    manifest = cli.write_basis({"d01": coastal}, path)
    with np.load(path, allow_pickle=False) as basis:
        expanded = cli.expand_from_basis(basis, manifest, {"d01": {name: 1.1 for name in names}})["d01"]
    assert all(expanded[name].tobytes() == changed[name].tobytes() for name in names)


def test_production_carry_context_validates_without_changing_static_holder(nl, tmp_path, research_w2_policy):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.surface_params import cli
    unseeded = OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True,
        noahmp_static=nl.noahmp_static, noahmp_energy_params=nl.noahmp_energy_params,
        noahmp_rad_params=nl.noahmp_rad_params, noahmp_options=nl.noahmp_options)
    assert unseeded.noahmp_land is None
    with pytest.raises(ValueError, match="bound noahmp_land"):
        sc.prior_controls(unseeded, ["smcmax"], device=False)
    values = sc.prior_controls(unseeded, ["smcmax"], device=False, land_state=nl.noahmp_land)
    bound = sc.bind_domain_controls(unseeded, {"d01": values}, "d01", land_state=nl.noahmp_land)
    assert bound.noahmp_land is None and unseeded.noahmp_land is None
    assert bound.noahmp_static is unseeded.noahmp_static
    report = cli.write_basis({"d01": unseeded}, tmp_path / "production_basis.npz",
                             land_states={"d01": nl.noahmp_land})
    assert report["domains"]["d01"]["controls"]["smcmax"]["active"]
    wet = nl.noahmp_land.replace(smois=values["smcmax"] + np.asarray(.01, values["smcmax"].dtype))
    with pytest.raises(ValueError, match="soil_porosity"):
        sc.bind_domain_controls(unseeded, {"d01": values}, "d01", land_state=wet)
    assert unseeded.noahmp_land is None and unseeded.surface_controls is None


def test_legacy_controls_ignore_optional_land_state(nl, monkeypatch):
    import gpuwrf.surface_params.lsmparams as lsm
    def poison(*args, **kwargs):
        raise AssertionError("legacy controls must not inspect optional land_state")
    monkeypatch.setattr(lsm, "validation_namelist", poison)
    plain = build()[5]
    expected = sc.expand_controls(plain, {"z0mvt": 1.1}, device=False)
    actual = sc.expand_controls(plain, {"z0mvt": 1.1}, device=False, land_state=object())
    assert actual["z0mvt"].tobytes() == expected["z0mvt"].tobytes()
    sc.validate_controls(actual, plain, land_state=object())
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    production = OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True,
        noahmp_static=plain.noahmp_static, noahmp_energy_params=plain.noahmp_energy_params,
        noahmp_rad_params=plain.noahmp_rad_params)
    bound = sc.bind_domain_controls(production, {"d01": actual}, "d01", land_state=object())
    assert bound.noahmp_land is None and bound.noahmp_static is production.noahmp_static


def test_loader_passes_the_fresh_start_predicate_to_bind_domain_controls():
    """F4' (critic): the production seam must make the NOAHMP_INIT refusal live on fresh starts only.

    Evaluates the actual ``cold_start=`` expression of the single bind call in ``_load_domains``
    (AST, no model run): fresh -> True, resume -> False, branch -> False.
    """
    import ast
    import dataclasses
    import inspect
    import textwrap
    from types import SimpleNamespace
    from gpuwrf.integration import nested_pipeline as pipeline
    tree = ast.parse(textwrap.dedent(inspect.getsource(pipeline._load_domains)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "bind_domain_controls"]
    assert len(calls) == 1, "expected exactly one surface-controls bind in _load_domains"
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(keywords["land_state"]) == "carry.noahmp_land"
    assert "cold_start" in keywords, "loader must pass cold_start (default True would refuse resumes)"
    predicate = compile(ast.Expression(keywords["cold_start"]), "<cold_start>", "eval")
    fields = {field.name for field in dataclasses.fields(pipeline.NestedPipelineConfig)}
    assert {"resume_checkpoint", "branch_from"} <= fields

    def cold_start(**overrides):
        config = SimpleNamespace(**{"resume_checkpoint": None, "branch_from": None, **overrides})
        return eval(predicate, {"getattr": getattr, "bool": bool}, {"config": config})
    assert cold_start() is True
    assert cold_start(resume_checkpoint=Path("checkpoint")) is False
    assert cold_start(branch_from=Path("branch")) is False
