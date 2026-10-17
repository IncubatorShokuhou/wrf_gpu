"""Established full default-program identity probe, with lane-private scratch."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
probe = root/'proofs/s1_port/release_identity.py'
text = probe.read_text().replace('<USER_HOME>/wrf_gpu2_lanes/s1-port',
                                 '<USER_HOME>/wrf_gpu2_lanes/t-godperf')
exec(compile(text,str(probe),'exec'),{'__file__':str(probe),'__name__':'__main__'})
