"""Run the established o1-nlbind Swiss identity probe in this lane's artifact root.

usage: python probe_release.py SOURCE OUT_JSON release
"""
from pathlib import Path
import sys

source = Path(sys.argv[1])
probe = source / 'proofs/o1_nlbind/probe.py'
text = probe.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                                'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/s1-dfi12")')
sys.path.insert(0, str(probe.parent))
exec(compile(text, str(probe), 'exec'), {'__file__': str(probe), '__name__': '__main__'})
