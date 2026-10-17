"""Run the established Swiss CLI program probe with lane-local scratch."""
from pathlib import Path
import sys

# Same policy as the merged s1_port release probe: CPU initialization, original
# CUDA Pallas callable during graph tracing. Actual JAX devices remain CPU.
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
try:
    from gpuwrf.backends import pallas as facade
except ImportError:
    pass
else:
    from gpuwrf.integration import nested_pipeline as pipeline
    original_load=pipeline._load_domains
    def load_on_cpu(*args,**kwargs):
        facade.cuda_native=lambda:False
        try:return original_load(*args,**kwargs)
        finally:facade.cuda_native=lambda:True
    pipeline._load_domains=load_on_cpu

p=Path(__file__).resolve().parents[2]/'o1_nlbind/probe.py'
source=p.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
    'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-bepbem")')
exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
