"""Frozen release probe, lane-local scratch; Swiss or O1_CASE=PROD d01."""
from pathlib import Path

probe = Path(__file__).resolve().parents[1] / "o1_nlbind/probe.py"
source = probe.read_text()
old = 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")'
assert source.count(old) == 1
source = source.replace(old, 'LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-streams")')
source = source.replace('ns, carry = bundles["d01"].namelist, carries["d01"]',
                        'ns, carry = bundles["d01"].namelist, carries["d01"]\n        assert carry.regional_streams is None')
exec(compile(source,str(probe),'exec'),{"__name__":"__main__","__file__":str(probe)})
