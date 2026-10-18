"""Run a script with the ALISIOS open-layer backstop (copied from lane w41-knownred): refused opens fail the run."""
import runpy, sys
from gpuwrf.assimilation.letkf.member_driver import refuse_protected_opens
with refuse_protected_opens() as refused:
    sys.argv = [sys.argv[1], *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name='__main__')
assert not refused, refused
