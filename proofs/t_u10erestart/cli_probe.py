"""Assert the loader's effective work before reusing the real CLI crash probe."""
import json
import os
from pathlib import Path
import runpy

assert os.environ['JAX_PLATFORMS'] == 'cpu'
from gpuwrf.integration import nested_pipeline as pipeline

load = pipeline._load_domains
def checked(*args, **kwargs):
    result = load(*args, **kwargs)
    expected = int(os.environ['TRF_EXPECTED_SURFACE'])
    work = {}
    for name, bundle in result[1].items():
        ns = bundle.namelist
        assert ns.sf_sfclay_physics == ns.bl_pbl_physics == expected
        work[name] = dict(sf_sfclay_physics=ns.sf_sfclay_physics,
                          bl_pbl_physics=ns.bl_pbl_physics,
                          mp_physics=ns.mp_physics, cu_physics=ns.cu_physics,
                          dt_s=ns.dt_s,
                          shape=[ns.grid.nz, ns.grid.ny, ns.grid.nx])
    path = Path(os.environ['TRF_ROOT']) / (os.environ['TRF_ACTION'] + '_effective_work.json')
    path.write_text(json.dumps(work, indent=2) + '\n')
    return result
pipeline._load_domains = checked
runpy.run_path(str(Path(__file__).parents[1] / 't_restartfix/cli_probe.py'), run_name='__main__')
