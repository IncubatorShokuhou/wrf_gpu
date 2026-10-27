"""Established release identity probe, with scratch files confined to this lane."""
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
probe = root / "proofs/s1_port/release_identity.py"
source = probe.read_text()
assert source.count('<USER_HOME>/wrf_gpu2_lanes/s1-port') == 1
source = source.replace('<USER_HOME>/wrf_gpu2_lanes/s1-port',
                        '<USER_HOME>/wrf_gpu2_lanes/w44-lutab')
exec(compile(source, str(probe), 'exec'),
     {'__file__': str(probe), '__name__': '__main__'})
