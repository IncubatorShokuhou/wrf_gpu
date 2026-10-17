"""Build hooks only; all project metadata lives in pyproject.toml.

Copies the small out-of-package runtime files (``bundle`` entries of
src/gpuwrf/assets_manifest.json) into the wheel as ``gpuwrf/_bundled/<repo path>``
after checking their pinned SHA-256, adds them to the sdist, and refuses a wheel
that would carry any file larger than 10 MiB (those tables are pinned ``remote``
release assets fetched on first use, see src/gpuwrf/assets.py).
"""
import hashlib
import json
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "src" / "gpuwrf" / "assets_manifest.json"


def _bundle():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return data["max_bundle_bytes"], {p: i for p, i in data["assets"].items() if i["kind"] == "bundle"}


class BuildPy(build_py):
    def run(self):
        super().run()
        if getattr(self, "editable_mode", False):
            return  # editable installs read the repo copies in place
        limit, bundle = _bundle()
        for repo_path, info in sorted(bundle.items()):
            source = ROOT / repo_path
            if not source.is_file():
                raise SystemExit(f"gpuwrf build: bundled runtime file {repo_path} is missing from the source tree")
            if hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]:
                raise SystemExit(f"gpuwrf build: {repo_path} differs from its pin; run scripts/update_assets_manifest.py")
            target = Path(self.build_lib) / "gpuwrf" / "_bundled" / repo_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        big = sorted(str(p) for p in (Path(self.build_lib) / "gpuwrf").rglob("*") if p.is_file() and p.stat().st_size > limit)
        if big:
            raise SystemExit(f"gpuwrf build: files larger than {limit} B must be pinned remote assets, not wheel data: {big}")


class Sdist(sdist):
    def add_defaults(self):
        super().add_defaults()
        self.filelist.extend(sorted(_bundle()[1]))


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
