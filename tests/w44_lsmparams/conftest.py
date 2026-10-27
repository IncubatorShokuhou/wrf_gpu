"""Explicit retained research tests; never a product admission switch."""
import pytest

_RESEARCH_W2_CASES = frozenset({
    ('test_admission', 'test_prior_is_bound_consumer_and_rejects_nonfinite'),
    ('test_admission', 'test_rsmin_requires_jarvis_and_soil_requires_table_schaake'),
    ('test_admission', 'test_soil_controls_refuse_unwired_optional_parameter_copies'),
    ('test_admission', 'test_changed_frozen_init_partition_refused_but_identity_and_retained_allowed'),
    ('test_admission', 'test_hvb_must_not_exceed_hvt'),
    ('test_admission', 'test_soil_class_mapping_uses_isltyp_not_vegetation'),
    ('test_admission', 'test_porosity_edit_rechecks_existing_moisture_via_shared_validator'),
    ('test_admission', 'test_all_wrf_urban_classes_preserve_fixed_porosity'),
    ('test_admission', 'test_glacier_partition_subsets_every_w2_field_only'),
    ('test_admission', 'test_cli_basis_lists_w2_and_preserves_soil_class_axis'),
    ('test_admission', 'test_retained_branch_soil_is_checked_without_mutating_original_holder'),
    ('test_admission', 'test_water_placeholders_are_preserved_without_allowing_zero_on_land'),
    ('test_admission', 'test_production_carry_context_validates_without_changing_static_holder'),
    ('test_checkpoint_cap', 'test_parameter_only_branch_refuses_retained_water_above_lower_cap'),
    ('test_checkpoint_cap', 'test_controlled_soil_only_check_preserves_retained_atmospheric_supersaturation'),
    ('test_checkpoint_cap', 'test_joint_state_and_porosity_edit_uses_effective_cap'),
})


@pytest.fixture
def research_w2_policy(monkeypatch, request):
    from gpuwrf.surface_params import controls as sc
    key = (request.module.__name__.rsplit(".", 1)[-1], request.node.originalname)
    assert key in _RESEARCH_W2_CASES
    monkeypatch.setattr(sc, "UNAVAILABLE_SURFACE_CONTROLS", frozenset())
