"""pytest plugin (-p mutant_overlay): load gpuwrf.assimilation.letkf submodules from $W41_MUTANT_DIR first.

Lazy (session start, after the repository conftests ran their CPU guard): the package path gets the mutant
directory first and already-imported letkf submodules are dropped, so test modules import the mutated code."""
import os
import sys


def pytest_sessionstart(session):
    directory = os.environ.get("W41_MUTANT_DIR")
    if not directory:
        return
    import gpuwrf.assimilation.letkf as pkg
    pkg.__path__.insert(0, directory)
    for name in [m for m in sys.modules if m.startswith("gpuwrf.assimilation.letkf.")]:
        del sys.modules[name]
