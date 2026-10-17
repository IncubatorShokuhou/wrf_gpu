"""Add diagnostic writes to a private FLG copy; removal must restore exact source.

Unit 93 is opened only by the lane's oracle driver. No physical operand, control
flow or argument is changed. Compare final native outputs to the unchanged build.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def instrument(path: Path) -> None:
    original = path.read_text()
    markers = [
        ("        call rad_all (    kmax-1, kmax,", """! BEGIN GPUWRF_FLG_SAVEPOINT
        write(93) kmax, as, u0ij, solcon, tsij, ee, pij, tij, qij, o3ij, &
          piwc, pde, plwc, pre, prwc, pgwc, cldamnt, cc_inho
! END GPUWRF_FLG_SAVEPOINT
"""),
        ("!C---- change by Yu for new trace gases\n        if (ngas.eq.0) then", """! BEGIN GPUWRF_FLG_SAVEPOINT
        write(93) nv, ib, dz, trp, ti, wi, wwi, tw, ww, www
        if (ib.gt.mbs) write(93) bf, bs
! END GPUWRF_FLG_SAVEPOINT
"""),
        ("!c 11/4/95 (begin)\n                  if ( ib .le. mbs ) then", """! BEGIN GPUWRF_FLG_SAVEPOINT
                  write(93) nv, ib, ig1, area, as(ib), u0, f0
                  if (ib.gt.mbs) write(93) ee(ib-mbs), bf, bs
                  write(93) wc1, wc2, wc3, wc4, wc, tt
! END GPUWRF_FLG_SAVEPOINT
"""),
        ("                end do\n              end do\n            end do\n31        end do", """! BEGIN GPUWRF_FLG_SAVEPOINT
                  write(93) fu1, fd1
! END GPUWRF_FLG_SAVEPOINT
"""),
    ]
    modified = original
    for needle, dump in markers:
        if modified.count(needle) != 1:
            raise ValueError(f"Capture marker count != 1: {needle!r}")
        modified = modified.replace(needle, dump + needle)
    restored = modified
    for _, dump in markers:
        restored = restored.replace(dump, "")
    assert restored == original
    path.write_text(modified)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    path.with_suffix(".instrumentation.json").write_text(json.dumps({
        "original_sha256": digest(original), "instrumented_sha256": digest(modified),
        "removed_writes_sha256": digest(restored), "removed_writes_restore_original": True,
        "scope": "Only native oracle; final outputs require byte-identity to unchanged FG01",
    }, indent=2) + "\n")


if __name__ == "__main__":
    instrument(Path(sys.argv[1]))
