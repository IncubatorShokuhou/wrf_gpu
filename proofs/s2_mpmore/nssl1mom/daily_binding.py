"""Exercise the actual daily loader on the real Swiss case for legacy19/21."""
from pathlib import Path
import json, os, re, sys, tempfile

tree, output, mp = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
assert mp in (19, 21) and os.environ['JAX_PLATFORMS'] == 'cpu'
assert (tree / '.git').is_file(), 'Use an immutable detached worktree'
sys.path.insert(0, str(tree / 'src'))
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.namelist_binding import require_namelist_honoured

lane = Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore')
reference = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
with tempfile.TemporaryDirectory(dir=lane) as tmp:
    case = Path(tmp) / 'case'
    case.mkdir()
    for item in reference.iterdir():
        if item.name != 'namelist.input':
            (case / item.name).symlink_to(item)
    text = re.sub(r'^\s*mp_physics\s*=.*$', f' mp_physics = {mp},',
                  (reference / 'namelist.input').read_text(), flags=re.M | re.I)
    (case / 'namelist.input').write_text(text)
    require_namelist_honoured(Gen2Run(case).namelist, ('d01',), replay=True)
    daily, _ = _build_real_case(DailyPipelineConfig(
        run_id=str(case), domain='d01', hours=1,
        output_dir=Path(tmp) / 'out', proof_dir=Path(tmp) / 'proof'))
    assert daily.namelist.mp_physics == mp
    result = dict(requested_mp=mp, bound_mp=daily.namelist.mp_physics,
                  grid=[daily.grid.nz, daily.grid.ny, daily.grid.nx], CPU=True,
                  scope='Actual daily initialization/binding; no executed forecast')
output.write_text(json.dumps(result, indent=2) + '\n')
print(result)
