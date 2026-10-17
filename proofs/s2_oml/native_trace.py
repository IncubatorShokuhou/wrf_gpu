"""Native CLI admission for OML and all hurricane surface-layer interfaces.

Uses the established o1 binding probe and s1_port facade policy: CPU loader,
native facade at trace, no GPU execution. Temporary case directories stay here.
"""
from pathlib import Path
import json
import sys

root=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2]).resolve()
sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline

original=pipeline._load_domains


def load_on_cpu(*a,**kw):
    facade.cuda_native=lambda:False
    try:return original(*a,**kw)
    finally:facade.cuda_native=lambda:True


pipeline._load_domains=load_on_cpu
probe=root/'proofs/o1_nlbind/probe.py'
source=probe.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',f'LANE = Path({str(output.parent)!r})')
source=source.replace('FIELDS = (','FIELDS = ("sf_ocean_physics", "isftcflx", "oml_hml0", "oml_gamma", "oml_relaxation_time", ',1)
arms=sys.argv[3:] or [
    'release',
    'sf_ocean_physics=1',
    'sf_ocean_physics=1,isftcflx=1',
    'sf_ocean_physics=1,isftcflx=2',
    'sf_ocean_physics=1,isftcflx=2,sf_sfclay_physics=1',
    'sf_ocean_physics=1,isftcflx=2,sf_sfclay_physics=91',
]
sys.argv=[str(probe),str(root),str(output),*arms]
exec(compile(source,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
result=json.loads(output.read_text())
result['native_facade_at_trace']=True
result['gpu_executed']=False
result['scope']='CLI admission + native tracing, not GPU execution or coupled-fidelity validation'
output.write_text(json.dumps(result,indent=2)+'\n')
assert all('sha' in record for record in result['arms'].values()),result
if 'release' in result['arms']:
    assert result['arms']['release']['sha'].startswith('2bf12cb8'),result['arms']['release']
