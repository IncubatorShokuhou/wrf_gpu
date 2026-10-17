from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from gpuwrf.config.paths import wrf_root


def test_rrtmg_fixture_generation_records_linked_harness_binary(tmp_path):
    # Exercise the real generator and original WRF objects in a private tree;
    # regeneration must never replace the retained numerical reference files.
    repo = Path(__file__).resolve().parents[1]
    original = Path(os.environ.get("WRF_ROOT", str(wrf_root()))).expanduser()
    build = Path(os.environ.get("WRF_BUILD", str(original / "_build_gen2_dmpar"))).expanduser()
    modules = ("module_ra_rrtmg_sw", "module_ra_rrtmg_lw")
    gen2_objects = [build / "CMakeFiles/WRF_Core.dir/phys" / f"{name}.F.o" for name in modules]
    pristine = Path(os.environ.get("WRF_PRISTINE_ROOT", str(original))).expanduser()
    phys = Path(os.environ.get("WRF_PRISTINE_PHYS", str(pristine / "phys"))).expanduser()
    if not all(path.is_file() for path in gen2_objects) and not all(
            (phys / f"{name}.o").is_file() for name in modules):
        pytest.skip(f"External original WRF RRTMG objects unavailable: {build}, {phys}")
    runtime = (original / "install_gen2_dmpar/run" if all(path.is_file() for path in gen2_objects)
               else Path(os.environ.get("WRF_PRISTINE_RUN", str(pristine / "run"))))
    for name in ("RRTMG_SW_DATA", "RRTMG_LW_DATA"):
        if not (runtime / name).is_file() and not (original / "run" / name).is_file():
            pytest.skip(f"External original WRF table unavailable: {runtime / name}")
    # Match the builder's GNU lookup, including the pristine override and
    # environment-directory fallback. Its -x checks are relative to build cwd.
    def executable(value):
        return bool(value) and os.access(tmp_path / value, os.X_OK)

    compiler = os.environ.get("FC", "")
    pristine_compiler = os.environ.get("WRF_PRISTINE_FC") or "gfortran"
    if not all(path.is_file() for path in gen2_objects) and executable(pristine_compiler):
        compiler = pristine_compiler
    wrf_env = os.environ.get("WRF_ENV") or os.environ.get("GPUWRF_WRF_ENV", "")
    if not compiler and wrf_env and executable(str(Path(wrf_env) / "bin/gfortran")):
        compiler = str(Path(wrf_env) / "bin/gfortran")
    if not compiler or not executable(compiler):
        compiler = shutil.which("gfortran")
    if not compiler:
        pytest.skip("Original WRF RRTMG harness requires an available Fortran compiler")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("m5_generate_rrtmg_fixture.py", "wrf_rrtmg_harness_build.sh",
                 "wrf_rrtmg_harness.f90", "extract_rrtmg_tables.py"):
        shutil.copy2(repo / "scripts" / name, scripts / name)
    (tmp_path / "src").symlink_to(repo / "src", target_is_directory=True)
    tables = tmp_path / "data" / "fixtures" / "rrtmg-tables-v1.npz"
    tables.parent.mkdir(parents=True)
    tables.symlink_to(repo / "data" / "fixtures" / tables.name)
    git_dir = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--absolute-git-dir"], text=True).strip()
    subprocess.run([sys.executable, str(scripts / "m5_generate_rrtmg_fixture.py")],
                   cwd=tmp_path, env={**os.environ, "GIT_DIR": git_dir},
                   check=True, timeout=180)
    manifest = yaml.safe_load((tmp_path / "fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["files"] if entry["path"] == "data/scratch/wrf_rrtmg_harness"]
    assert entries
    harness = tmp_path / entries[0]["path"]
    assert harness.exists()
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == entries[0]["checksum_sha256"]
    # The supported pristine GNU fallback uses .o, not Gen2's .F.o names.
    for name, obj in zip(modules, gen2_objects):
        assert f"{name}.F.o={'present' if obj.is_file() else 'absent'}" in manifest["source_commit"]
    linked = dict(line.split("=", 1) for line in
                  (tmp_path / "data/scratch/wrf_rrtmg_harness_build.log").read_text().splitlines()
                  if line.startswith(("real_sw_object=", "real_lw_object=")))
    assert Path(linked["real_sw_object"]).is_file()
    assert Path(linked["real_lw_object"]).is_file()
    symbols = subprocess.check_output(["nm", "-g", str(harness)], text=True, timeout=30).lower()
    assert "rrtmg_swrad" in symbols and "rrtmg_lwrad" in symbols
    assert "full_rrtmg_driver_call=RRTMG_SWRAD+RRTMG_LWRAD" in manifest["source_commit"]
