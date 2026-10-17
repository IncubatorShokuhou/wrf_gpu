"""wz1080 box tooling: temp cleanup only through the confined delete helper (no bare `rm -rf "$VAR"`)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REMOTE = REPO / "scripts" / "remote"
COMMON = REMOTE / "wz1080_common.sh"
SYNC_RE = r"^sync\.[A-Za-z0-9]{8}$"


def _confined(root, path, name_re=SYNC_RE, *, record=None):
    """Run wz_rm_confined in bash; with `record`, rm is replaced by a recorder (a broken guard deletes nothing)."""
    shim = f'rm() {{ printf "%s\\n" "$*" >> "{record}"; }}; ' if record else ""
    script = f'source "{COMMON}"; {shim}wz_rm_confined "$1" "$2" "$3"'
    return subprocess.run(["bash", "-c", script, "wz", str(root), str(path), name_re],
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "lane" / "bundles"
    r.mkdir(parents=True)
    return r


def test_mktemp_child_is_removed_recursively(root):
    tmpd = root / "sync.Ab3dE6gH"
    (tmpd / "r.git" / "objects" / "info").mkdir(parents=True)
    (tmpd / "x.bundle").write_bytes(b"bundle")
    keep = root / "other.txt"
    keep.write_text("keep")
    res = _confined(root, tmpd)
    assert res.returncode == 0, res.stderr
    assert not tmpd.exists() and keep.exists() and root.is_dir()


def test_missing_child_is_a_noop(root):
    assert _confined(root, root / "sync.Ab3dE6gH").returncode == 0


@pytest.mark.parametrize("case", ["empty_path", "empty_root", "root_itself", "outside_root", "nested",
                                  "dotdot", "wrong_name", "symlink_out", "noncanonical_root", "shallow_root"])
def test_everything_else_is_refused_and_untouched(case, root, tmp_path):
    record = tmp_path / "rm_calls.log"
    outside = tmp_path / "outside"
    (outside / "sync.Ab3dE6gH").mkdir(parents=True)
    child = root / "sync.Ab3dE6gH"
    child.mkdir()
    (root / "keepme").mkdir()
    (child / "inner").mkdir()
    args = {
        "empty_path": (root, ""),
        "empty_root": ("", child),
        "root_itself": (root, root),
        "outside_root": (root, outside / "sync.Ab3dE6gH"),
        "nested": (root, child / "inner"),
        "dotdot": (root, f"{child}/../sync.Ab3dE6gH"),
        "wrong_name": (root, root / "keepme"),
        "symlink_out": (root, root / "sync.Zz9yY8xX"),
        "noncanonical_root": (f"{root}/.", child),
        "shallow_root": ("/tmp", "/tmp/sync.Ab3dE6gH"),
    }[case]
    if case == "symlink_out":
        (root / "sync.Zz9yY8xX").symlink_to(outside / "sync.Ab3dE6gH", target_is_directory=True)
    if case == "shallow_root" and not Path("/tmp/sync.Ab3dE6gH").exists():
        Path("/tmp/sync.Ab3dE6gH").mkdir()
        created_tmp = True
    else:
        created_tmp = False
    try:
        res = _confined(*args, record=record)
        assert res.returncode == 3 and "refuse delete" in res.stderr, (res.returncode, res.stderr)
        assert not record.exists(), record.read_text()
    finally:
        if created_tmp:
            Path("/tmp/sync.Ab3dE6gH").rmdir()
    assert child.is_dir() and (outside / "sync.Ab3dE6gH").is_dir() and (root / "keepme").is_dir()


def test_remote_scripts_have_no_bare_variable_recursive_delete():
    """Every recursive delete in scripts/remote goes through wz_rm_confined; sync uses it for its temp dir."""
    offenders = []
    for path in sorted(REMOTE.rglob("*")):
        if not path.is_file() or path.suffix not in {".sh", ".py", ""}:
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"\brm\s+-[a-zA-Z]*r", line) and 'rm -rf -- "$rpath"' not in line:
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
            if re.search(r"(rmtree|shutil\.rmtree)", line):
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offenders, offenders
    sync = (REMOTE / "wz1080_sync.sh").read_text()
    assert 'mktemp -d "$BUNDLE_ROOT/sync.XXXXXXXX"' in sync
    assert sync.count('wz_rm_confined "$BUNDLE_ROOT" "$TMPD" "$TMPD_RE"') == 2
