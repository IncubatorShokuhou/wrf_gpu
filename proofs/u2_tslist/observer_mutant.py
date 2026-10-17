"""Genuine deletion mutant: drop only the post-solve station observer hook."""
def pytest_runtest_setup(item):
    if item.name!='test_hook_runs_after_physics_and_off_has_no_observer':
        return
    from gpuwrf.runtime import operational_mode as om
    def without_observer(carry,namelist,step_index,**kwargs):
        return om._physics_boundary_step_with_limiter_diagnostics(carry,namelist,step_index,**kwargs)[0]
    om._physics_boundary_step=without_observer
