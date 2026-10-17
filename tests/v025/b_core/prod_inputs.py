"""Fresh PROD d01+d02 inputs for b-core CPU tests (replaces integrate's common_inputs.pkl, deleted in the
2026-10-03 disk cleanup): the product loader nested_pipeline._load_domains, once per process, with the
State GPU guard pointed at the CPU device for the duration of the load only (conftest's GPU-skip hook
keeps working for every other test).  Skips when the PROD inputs are absent."""
import functools
from pathlib import Path

import pytest

PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")


@functools.lru_cache(maxsize=1)
def _load(names=("d01", "d02")):
    import jax

    import gpuwrf.contracts.state as state_module
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains

    assert jax.devices()[0].platform == "cpu"
    guard = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices()[0]
    try:
        # _load_domains reads only input_dir/feedback from the config; the output paths are never touched.
        config = NestedPipelineConfig(PROD, PROD / "unused_out", PROD / "unused_proof", hours=1, max_dom=2)
        loaded = _load_domains(config, names)
        if names == ("d01",):
            # _load_domains seeds this optional leaf only for len(names)>1.
            # Preserve the nested PROD root's carry ABI while omitting d02 work.
            from gpuwrf.nesting.nest_o3 import nest_o3_from_parent_enabled
            import jax.numpy as jnp

            carry = loaded[5]["d01"]
            if carry.radiation_diagnostics is not None and nest_o3_from_parent_enabled():
                loaded[5]["d01"] = carry.replace(
                    o3rad=jnp.zeros(carry.state.theta.shape, jnp.float32)
                )
        return loaded
    finally:
        state_module._gpu_device = guard


def prod_domains(*, root_only=False):
    """The old pickle tuple; root-only structural gates can omit unused child init."""
    names = ("d01",) if root_only else ("d01", "d02")
    if not (PROD / f"wrfinput_{names[-1]}").exists():
        pytest.skip(f"PROD inputs unavailable: {PROD}")
    return _load(names)
