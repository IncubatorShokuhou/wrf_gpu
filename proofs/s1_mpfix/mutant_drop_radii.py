"""Delete the production WSM7 radius handoff in a disposable detached tree."""
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
path=root/'src/gpuwrf/coupling/scan_adapters.py'
s=path.read_text()
block='''    # WSM7 produces WRF radiation radii as diagnostics. Hold them only when
    # physics_init's resolved selection allocated the radius carry.
    for name in ("re_cloud", "re_ice", "re_snow"):
        if getattr(state, name) is not None and name in tendency.diagnostics:
            updates[name] = _back3d(tendency.diagnostics[name]).astype(jnp.float32)
'''
assert s.count(block)==1
path.write_text(s.replace(block,''))
