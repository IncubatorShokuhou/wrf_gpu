"""Explicit test-only policy for retained research/component assertions.

Never autouse; only the listed test/module pairs may request this fixture.
Release admission tests retain the real refusal policy. No product switch.
"""
import pytest

_RESEARCH_DIFFUSION_CASES = frozenset({
    ('test_checkpoint_consumers', 'test_branch_static_view_excludes_only_declared_operands'),
    ('test_checkpoint_consumers', 'test_pickle_consumer_authenticates_saved_and_requested_values'),
    ('test_checkpoint_consumers', 'test_restart_identity_records_each_domain_family'),
    ('test_production_callers', 'test_acoustic_default_hardcode_is_detected'),
    ('test_production_callers', 'test_acoustic_scan_forwards_native_damping'),
    ('test_production_callers', 'test_acoustic_stage_forwards_each_vertical_cfl_control'),
    ('test_production_callers', 'test_each_sixth_order_production_site_forwards_factor'),
    ('test_production_callers', 'test_individual_sixth_order_callsite_deletion_is_detected'),
    ('test_production_callers', 'test_vertical_cfl_caller_deletion_is_detected'),
    ('test_production_callers', 'test_vertical_cfl_threshold_mapping_deletion_is_detected'),
    ('test_runtime_parameters', 'test_bad_values_refused'),
    ('test_runtime_parameters', 'test_changed_w_crit_cfl_pristine_wrf_and_single_compile'),
    ('test_runtime_parameters', 'test_family_roundtrip_keys_and_digest'),
    ('test_runtime_parameters', 'test_inactive_consumers_refused'),
    ('test_runtime_parameters', 'test_two_sixth_order_values_one_compiled_program'),
    ('test_smag_c_s', 'test_actual_diffusion_caller_forwards_c_s'),
    ('test_smag_c_s', 'test_c_s_admission_and_key'),
})


@pytest.fixture
def research_diffusion_policy(monkeypatch, request):
    from gpuwrf import runtime_parameters as params
    key = (request.module.__name__.rsplit(".", 1)[-1], request.node.originalname)
    assert key in _RESEARCH_DIFFUSION_CASES
    monkeypatch.setattr(params, "UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES",
                        params.UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES - {"diffusion_parameters"})
