"""Link read-only case assets; create mutable namelists exclusively in the lane."""
import re
from pathlib import Path

MUTABLE = re.compile(r"^(?:namelist\.|wrfout|wrfrst|rsl\.|fort\.|d[01]$|(?:wrf|real|ideal|ndown|tc)\.exe$)")


def link_assets(seed: Path, case: Path, *, lane_root: Path, exclude=()):
    if not case.resolve().is_relative_to(lane_root.resolve()):
        raise ValueError("case must be inside the lane artifact directory")
    for source in seed.iterdir():
        if source.is_symlink() and source.name not in exclude and not MUTABLE.match(source.name):
            (case / source.name).symlink_to(source.resolve())


def write_namelist(case: Path, data: bytes, *, lane_root: Path):
    if not case.resolve().is_relative_to(lane_root.resolve()):
        raise ValueError("case must be inside the lane artifact directory")
    # O_EXCL refuses an existing symlink, including one whose target is absent.
    with (case / "namelist.input").open("xb") as stream:
        stream.write(data)
