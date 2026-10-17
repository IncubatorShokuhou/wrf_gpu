"""Run the frozen release identity tool, confining temporary files to this lane."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Since s1-port the local facade deliberately interprets native kernels on
# CPU. Match its reviewed release_identity.py protocol: interpret init, then
# trace the original CUDA callable without ever executing a GPU command.
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
original_load=pipeline._load_domains
def load_cpu_then_trace_native(*args,**kwargs):
    facade.cuda_native=lambda:False
    try:
        return original_load(*args,**kwargs)
    finally:
        facade.cuda_native=lambda:True
pipeline._load_domains=load_cpu_then_trace_native
probe = ROOT / "proofs/o1_nlbind/probe.py"
text = probe.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
new = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-twoway/trace-case")'
assert text.count(old) == 1
text = text.replace(old, new)
needle='        text = canon(jx)'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n        Path(OUT).with_suffix(".jaxpr.txt").write_text(text)')
Path("<USER_HOME>/wrf_gpu2_lanes/u2-twoway/trace-case").mkdir(exist_ok=True)
sys.argv = [str(probe), str(ROOT), sys.argv[1], "release"]
exec(compile(text, str(probe), "exec"), {"__name__":"__main__", "__file__":str(probe)})
