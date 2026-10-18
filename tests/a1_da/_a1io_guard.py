"""Guarded ALISIOS reads for a1-io real-file tests (binding ALISIOS rule, wrf_gpu2 manager 2026-10-17 01:25Z).

``module_guard(read_set_fn)`` returns an autouse module fixture by default. Mixed synthetic/real modules
use ``autouse=False`` and request it only for real-data tests. The admitted tests run inside ONE
``alisios_guard.guarded_session`` over exactly the listed {path: [valid days]} (guard scan refuses DENY days before
anything is opened; the read spy refuses unlisted ALISIOS paths). Empty read set (data absent) → no session.
"""
from __future__ import annotations

import pytest

from gpuwrf.assimilation.letkf import alisios_guard as ag
from gpuwrf.assimilation.letkf import archive_ensemble as ae


def module_guard(read_set_fn, *, autouse=True):
    @pytest.fixture(scope="module", autouse=autouse)
    def _alisios_guard(tmp_path_factory):
        read_set = {str(p): d for p, d in read_set_fn().items()}
        if not read_set:
            yield None
            return
        issues = sorted({m.group(1) for p in read_set if (m := ae._RUN_RE.search(p))})
        with ag.guarded_session(read_set, issues, tmp_path_factory.mktemp("alisios_guard"),
                                purpose="a1-io real-file tests (A1 LETKF DA probe)") as session:
            yield session
        assert not session.receipt["spy"]["refused_paths"], session.receipt["spy"]["refused_paths"]
    return _alisios_guard
