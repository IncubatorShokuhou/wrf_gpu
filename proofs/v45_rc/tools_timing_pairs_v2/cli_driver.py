"""Run the real CLI; assert backend inside its measured forecast process.

SEQ_SMOKE bounds only the actual root-step count after checking the original
requested forecast duration. GPU production uses every original model step.
"""
import json,os,sys
from pathlib import Path
from gpuwrf import cli
assert Path(cli.__file__).resolve().is_relative_to(Path(os.environ["PYTHONPATH"]).resolve()),cli.__file__
args=cli.build_parser().parse_args()
from alignment import check
check(args)
if bool(getattr(args,'dry_run',False)):
    raise SystemExit(args.func(args))
# Product allocator/C-auto setup must precede every JAX backend initialization.
cli._maybe_reexec_for_nested_allocator(args)
from gpuwrf.integration import nested_pipeline as pipeline
import jax
if os.environ.get('W45R_PROFILE')=='1':
    from profile_hooks import install
    install(pipeline,jax)
original=pipeline._execute_nested_pipeline
smoke=os.environ.get('WRF_GPU2_SEQ_SMOKE')=='1'
def execute(*pos,**kw):
    device=jax.devices()[0]
    expected='cpu' if smoke else 'gpu'
    assert device.platform==expected,(device,expected)
    Path('device_measured.json').write_text(json.dumps(dict(pid=os.getpid(),platform=device.platform,device=str(device),source_file=cli.__file__,CPU_smoke=smoke,scope='same process as the measured CLI model'),indent=2)+'\n')
    return original(*pos,**kw)
pipeline._execute_nested_pipeline=execute
if smoke:
    original_steps=pipeline._steps_by_domain
    def one_root(hours,dts,names):
        original=original_steps(hours,dts,names) # includes real duration-alignment checks
        root=names[0]
        bounded={n:int(round(float(dts[root])/float(dts[n]))) for n in names}
        assert all(original[n]>=bounded[n]>0 for n in names)
        Path('CPU_SMOKE_BOUNDED.json').write_text(json.dumps(dict(requested_hours=hours,full_steps=original,executed_steps=bounded,GPU_native_gates='NOT-APPLICABLE',science='not a fidelity or performance result'),indent=2)+'\n')
        return bounded
    pipeline._steps_by_domain=one_root
raise SystemExit(args.func(args))
