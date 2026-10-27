"""CPU launch admission only: real RRTMG caller on a 4x4x4 grid, not forecast truth."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def forecast_smoke():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.coupling import physics_couplers as pc
    assert os.environ['TOPO_LAUNCH_SMOKE'] == '1' and os.environ['JAX_PLATFORMS'] == 'cpu'
    assert jax.devices()[0].platform == 'cpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1'
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    started = time.perf_counter()
    fixture = Path(os.environ['BENCH_SRC_REPO']) / 'tests/test_rrtmg_topographic_coupling.py'
    spec = importlib.util.spec_from_file_location('topo_launch_mini_fixture', fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    terrain = jnp.asarray([[0, 500, 1000, 1500]] * 4, jnp.float32)
    grid = module._grid(terrain=terrain)
    # Use the production cold-init seam: MP_RE=1 requires its held radii.
    state = module._state(grid).ensure_conditional_leaves(
        mp_physics=8, use_mp_re=1, ra_lw_physics=4, ra_sw_physics=4)
    static = pc.build_radiation_static_from_wrf_fields(jnp.full((4, 4), 28.3, jnp.float32),
        jnp.full((4, 4), -16.4, jnp.float32), terrain, dx_m=3000., dy_m=3000.)
    setup = time.perf_counter() - started
    began = time.perf_counter()
    diagnostics = jax.block_until_ready(pc.rrtmg_radiation_diagnostics(state, grid,
        time_utc=datetime(2026, 2, 28, 8, tzinfo=timezone.utc), lead_seconds=0.,
        radiation_static=static, topo_shading=1, slope_rad=1))
    elapsed = time.perf_counter() - began
    leaves = [np.asarray(x) for x in jax.tree.leaves(diagnostics)]
    assert leaves and all(np.isfinite(x).all() for x in leaves)
    net = np.asarray(diagnostics.swnorm) - np.asarray(diagnostics.swup_topographic)
    assert net.shape == (4, 4) and np.isfinite(net).all()
    report = dict(status='CPU_LAUNCH_SMOKE_PASS', scientific_GPU_gate='PENDING', platform='cpu',
        shape=[4, 4, 4], model_module=pc.__file__, callable='rrtmg_radiation_diagnostics',
        controls=dict(ra_sw_physics=4, ra_lw_physics=4, topo_shading=1, slope_rad=1),
        finite_leaves=len(leaves), setup_s=setup, radiation_compile_first_call_s=elapsed,
        whole_smoke_s=time.perf_counter() - started)
    (out / 'LAUNCH_SMOKE_MODEL.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0)
