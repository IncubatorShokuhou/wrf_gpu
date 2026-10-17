"""Release tooling in scripts/release (private-only; skipped in the curated public tree). CPU, no network."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts" / "release"
if not (TOOLS / "curate.py").is_file():
    pytest.skip("release tooling is private-only (not in the curated public tree)", allow_module_level=True)


def _load(name):
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(f"_release_{name}", TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


curate = _load("curate")
verify = _load("verify")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_no_release_is_hard_coded():
    pins = ('default="0.3.4"', "'0.3.4'", '"0.3.4"', "v034-release-prep", "wrf_gpu2_lanes/rel034", "taskset -c 24,25",
            "curated v0.3.4")
    for path in TOOLS.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not [pin for pin in pins if pin in text], path.name
    for args in ([sys.executable, str(TOOLS / "verify.py"), "--pub", ".", "--private", "HEAD"],
                 [sys.executable, str(TOOLS / "curate.py"), "--private", "HEAD", "--out", ".", "--work", "w"],
                 ["bash", str(TOOLS / "final.sh"), "--private", "HEAD"]):
        rc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        assert rc.returncode == 2 and "version" in rc.stderr.lower(), args


def test_tooling_and_build_hook_layout():
    assert "setup.py" in curate.ROOT_KEEP  # the wheel build hook must reach the public root
    assert curate.PRIVATE_ONLY == ("scripts/release/",)
    committed = json.loads(_git(ROOT, "show", f"HEAD:{curate.ASSETS_MANIFEST}"))["assets"]
    assert curate.remote_assets("HEAD") == {p: i for p, i in committed.items() if i["kind"] == "remote"}


@pytest.fixture
def toy_repo(tmp_path, monkeypatch):
    """A private repo with one remote table and one bundle file pinned in the assets manifest."""
    repo = tmp_path / "priv"
    (repo / "src/gpuwrf/physics/scheme").mkdir(parents=True)
    (repo / "scripts").mkdir()
    big = b"T" * 4096
    (repo / "src/gpuwrf/physics/scheme/big.npz").write_bytes(big)
    (repo / "scripts/small.py").write_text("x = 1\n")
    manifest = dict(schema="gpuwrf-runtime-assets-v1", url_template="u", max_bundle_bytes=10, assets={
        "src/gpuwrf/physics/scheme/big.npz": dict(kind="remote", asset="scheme-big.npz", bytes=len(big),
                                                 sha256=hashlib.sha256(big).hexdigest()),
        "scripts/small.py": dict(kind="bundle", asset="small.py", bytes=6,
                                 sha256=hashlib.sha256(b"x = 1\n").hexdigest())})
    (repo / curate.ASSETS_MANIFEST).write_text(json.dumps(manifest))
    (repo / "setup.py").write_text("")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy")
    monkeypatch.setattr(curate, "PRIV_REPO", str(repo))
    return repo


def test_release_assets_are_exported_and_verified(toy_repo, tmp_path):
    remote = curate.remote_assets("HEAD")
    out = tmp_path / "assets"
    rows = curate.export_release_assets("HEAD", remote, out)
    assert set(rows) == {"scheme-big.npz"} and (out / "scheme-big.npz").read_bytes() == b"T" * 4096
    assert (out / "SHA256SUMS").read_text() == f"{rows['scheme-big.npz']['sha256']}  scheme-big.npz\n"
    pub = toy_repo  # a public tree carrying the same manifest
    good = verify.release_assets(pub, out)
    assert not good["missing"] and not good["bad"] and not good["unexpected"] and good["remote"] == 1
    (out / "scheme-big.npz").write_bytes(b"T" * 4095 + b"X")
    assert verify.release_assets(pub, out)["bad"] == ["scheme-big.npz"]
    (out / "scheme-big.npz").unlink()
    (out / "stray.bin").write_bytes(b"")
    bad = verify.release_assets(pub, out)
    assert bad["missing"] == ["scheme-big.npz"] and bad["unexpected"] == ["stray.bin"]
    assert verify.release_assets(pub, None)["missing"] == ["scheme-big.npz"]


def test_export_refuses_a_blob_that_differs_from_its_pin(toy_repo, tmp_path):
    remote = curate.remote_assets("HEAD")
    remote["src/gpuwrf/physics/scheme/big.npz"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="differs from its manifest pin"):
        curate.export_release_assets("HEAD", remote, tmp_path / "assets")


def test_packaging_gate_needs_setup_py_and_pinned_bundles(toy_repo):
    assert verify.packaging(toy_repo) == dict(setup_py=True, manifest=True, bundle_bad=[])
    (toy_repo / "scripts/small.py").write_text("x = 2\n")
    assert verify.packaging(toy_repo)["bundle_bad"] == ["scripts/small.py"]
    (toy_repo / "setup.py").unlink()
    assert verify.packaging(toy_repo)["setup_py"] is False


def test_tree_parity_reports_remote_tables_as_release_assets(toy_repo, tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    (pub / "src/gpuwrf").mkdir(parents=True)
    (pub / curate.ASSETS_MANIFEST).write_bytes((toy_repo / curate.ASSETS_MANIFEST).read_bytes())
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    rep = verify.tree_parity(pub, "HEAD", "src/gpuwrf", set(curate.remote_assets("HEAD")))
    assert rep["release_assets"] == ["src/gpuwrf/physics/scheme/big.npz"] and rep["missing"] == []
    rep = verify.tree_parity(pub, "HEAD", "src/gpuwrf", set())
    assert rep["missing"] == ["src/gpuwrf/physics/scheme/big.npz"]  # without the manifest it is a real gap


GATES = ("verify", "committed", "big", "uncommitted", "import", "collect", "help", "docsite", "wheel", "deptests")


@pytest.mark.parametrize("failing", GATES)
def test_final_gate_exits_nonzero_for_each_failed_gate(failing):
    gates = {name: 0 for name in GATES}
    ok = subprocess.run([sys.executable, str(TOOLS / "gate.py"), *(f"{k}={v}" for k, v in gates.items()),
                         "--info", "assets=20"], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.startswith("GATE PASS")
    gates[failing] = 1
    bad = subprocess.run([sys.executable, str(TOOLS / "gate.py"), *(f"{k}={v}" for k, v in gates.items())],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and f"failed={failing}" in bad.stdout


def test_final_sh_exit_status_is_the_gate_and_forwards_base():
    text = (TOOLS / "final.sh").read_text()
    tail = text[text.index('python3 "$TOOLS/gate.py"'):]
    assert 'exit "${PIPESTATUS[0]}"' in tail and tail.strip().endswith('exit "${PIPESTATUS[0]}"')
    for gate in ("verify=$VG", "committed=$CS", "big=$BIG", "uncommitted=", "import=$CI", "collect=$CC", "help=$CL",
                 "docsite=$CD", "wheel=$CW", "deptests=$DT"):
        assert gate in tail, gate
    curate_call = text[text.index('"$TOOLS/curate.py"'):text.index("> \"$R/curate.out\"")]
    assert '--base "$BASE"' in curate_call
    assert subprocess.run([sys.executable, str(TOOLS / "gate.py"), "verify="], capture_output=True).returncode == 1


def test_published_release_records_keep_their_public_bytes(toy_repo, tmp_path, monkeypatch):
    (toy_repo / "docs/release").mkdir(parents=True)
    (toy_repo / "release_notes").mkdir()
    old = b"# v0.3.4 support matrix (published)\n"
    (toy_repo / "docs/release/SUPPORT_MATRIX_v0.3.4.md").write_bytes(old)
    (toy_repo / "release_notes/RELEASE_NOTES_v0.3.4.md").write_bytes(b"notes\n")
    (toy_repo / "docs/release/SUPPORT_MATRIX_v0.4.0.md").write_bytes(b"new\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    assert curate.historical_records(curate.ls_tree("HEAD"), "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    # src/tests mirror the private commit (tree parity); a non-version token is not a release name
    assert curate.historical_records(["src/gpuwrf/x_v0.3.4.json", "tests/t_v0.3.4.py", "docs/x_v0.3.4x.md"], "0.4.0") == []
    pub = tmp_path / "pub"
    for rel in ("docs/release/SUPPORT_MATRIX_v0.3.4.md", "docs/release/SUPPORT_MATRIX_v0.4.0.md"):
        (pub / rel).parent.mkdir(parents=True, exist_ok=True)
    (pub / "docs/release/SUPPORT_MATRIX_v0.3.4.md").write_bytes(old + b"## appended v0.4 development section\n")
    (pub / "docs/release/SUPPORT_MATRIX_v0.4.0.md").write_bytes(b"new, edited\n")
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))  # verify.historical reads the base through verify.PRIV
    assert verify.historical(pub, "HEAD", "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    assert curate.restore_historical(pub, "HEAD", "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    assert (pub / "docs/release/SUPPORT_MATRIX_v0.3.4.md").read_bytes() == old
    assert (pub / "docs/release/SUPPORT_MATRIX_v0.4.0.md").read_bytes() == b"new, edited\n"  # current release
    assert verify.historical(pub, "HEAD", "0.4.0") == []


def test_no_remote_table_may_stay_in_the_public_tree(toy_repo, tmp_path):
    pub = tmp_path / "pub"
    (pub / "src/gpuwrf").mkdir(parents=True)
    (pub / curate.ASSETS_MANIFEST).write_bytes((toy_repo / curate.ASSETS_MANIFEST).read_bytes())
    assert verify.remote_in_tree(pub) == []
    (pub / "src/gpuwrf/physics/scheme").mkdir(parents=True)
    (pub / "src/gpuwrf/physics/scheme/big.npz").write_bytes(b"T")
    assert verify.remote_in_tree(pub) == ["src/gpuwrf/physics/scheme/big.npz"]
    curate_src = (TOOLS / "curate.py").read_text()
    assert "kept_identical_in_base" not in curate_src and "base_hash.get(repo_path)" not in curate_src


@dataclasses.dataclass(frozen=True)
class AotMeta:  # pickled under this module's name; compile_bundle reads it without importing gpuwrf
    hlo_sha256: str
    blob_sha256: str
    cheap_key: str
    fingerprint: dict
    in_tree: object = None


def _toy_cache(root, flags=""):
    cache = root / "jax"
    d01 = cache / "aot/0.4.0-cuda_sm120/d01"
    d01.mkdir(parents=True)
    blob = b"ELF-native-blob"
    (d01 / "k_a.xlaexec").write_bytes(blob)
    target = dict(platform="gpu", device_kind="NVIDIA GeForce RTX 5090", compute_capability="12.0", cuda_driver="13000",
                  jax_version="0.10.0", jaxlib_version="0.10.0", x64=True, xla_flags=flags)
    meta = AotMeta("h1", hashlib.sha256(blob).hexdigest(), "a", target, in_tree=range(3))
    (d01 / "k_a.meta").write_bytes(pickle.dumps(meta))
    (d01 / "last_cheap_key").write_text("a")
    (cache / "jit_f-00-cache").write_bytes(b"entry")
    (cache / "jit_f-00-atime").write_bytes(b"1")
    (cache / "xla_gpu_per_fusion_autotune_cache_dir").mkdir()
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("tune")
    return cache, blob, meta


def test_compile_bundle_seal_is_deterministic_and_check_detects_compiles_outside(toy_repo, tmp_path, capsys):
    cb = _load("compile_bundle")
    cache, blob, meta = _toy_cache(tmp_path)
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--source", str(toy_repo), "--repo", str(toy_repo)]
    assert cb.main(seal + ["--out", str(tmp_path / "b1")]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line == (f"TUPLE commit={_git(toy_repo, 'rev-parse', 'HEAD').strip()} "
                    f"src_tree={_git(toy_repo, 'rev-parse', 'HEAD:src').strip()} "
                    f"gpuwrf_tree={_git(toy_repo, 'rev-parse', 'HEAD:src/gpuwrf').strip()} "
                    f"bundle_sha256={hashlib.sha256((tmp_path / 'b1/gpuwrf-bundle.tar.gz').read_bytes()).hexdigest()}")
    # a second process with another hash seed writes the same bytes (bundle SHA = id, never run-dependent)
    subprocess.run([sys.executable, str(TOOLS / "compile_bundle.py"), *seal, "--out", str(tmp_path / "b2")], check=True,
                   capture_output=True, env={**os.environ, "PYTHONHASHSEED": "12345"})
    assert (tmp_path / "b1/gpuwrf-bundle.tar.gz").read_bytes() == (tmp_path / "b2/gpuwrf-bundle.tar.gz").read_bytes()
    manifest = json.loads((tmp_path / "b1/BUNDLE_MANIFEST.json").read_text())
    assert manifest["aot_meta"]["aot/0.4.0-cuda_sm120/d01/k_a.meta"]["hlo_sha256"] == "h1"
    assert manifest["counts"] == dict(aot_blob=1, aot_meta=1, jax_entry=1, mutable=2, tune=1)
    with tarfile.open(tmp_path / "b1/gpuwrf-bundle.tar.gz") as tar:
        names = tar.getnames()
    assert "gpuwrf-bundle/cache/jit_f-00-cache" in names and not [n for n in names if n.endswith("-atime")]
    check = ["check", "--cache", str(cache), "--manifest", str(tmp_path / "b1/BUNDLE_MANIFEST.json")]
    assert cb.main(check) == 0
    (cache / "jit_f-00-atime").write_bytes(b"2")  # access stamps and the prefetch hint may change
    (cache / "aot/0.4.0-cuda_sm120/d01/last_cheap_key").write_text("b")
    alias = cache / "aot/0.4.0-cuda_sm120/d01/k_b.xlaexec"
    alias.write_bytes(blob)  # cheap-key alias of the sealed executable
    (alias.with_suffix(".meta")).write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="b")))
    assert cb.main(check) == 0 and "aliases=2" in capsys.readouterr().out
    for orphan in ("k_c.meta", "k_d.xlaexec", "k_e"):  # half pairs and a foreign program never count as aliases
        d01 = cache / "aot/0.4.0-cuda_sm120/d01"
        if orphan == "k_c.meta":
            (d01 / orphan).write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="c")))
        elif orphan == "k_d.xlaexec":
            (d01 / orphan).write_bytes(blob)
        else:
            (d01 / "k_e.xlaexec").write_bytes(blob)
            (d01 / "k_e.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="e", hlo_sha256="other")))
        assert cb.main(check) == 1 and "BUNDLE_BROKEN" in capsys.readouterr().out, orphan
        for f in d01.glob(orphan if orphan != "k_e" else "k_e.*"):
            f.unlink()
    assert cb.main(check) == 0
    (cache / "jit_g-11-cache").write_bytes(b"new program")
    assert cb.main(check) == 1 and "new=1" in capsys.readouterr().out
    (cache / "jit_g-11-cache").unlink()
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("retuned")
    assert cb.main(check) == 1 and "changed=1" in capsys.readouterr().out
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("tune")
    assert cb.main(check) == 0
    shutil.copytree(cache, tmp_path / "moved")  # identical bytes elsewhere: JAX/AOT keys pin the absolute path
    moved = ["check", "--cache", str(tmp_path / "moved"), "--manifest", str(tmp_path / "b1/BUNDLE_MANIFEST.json")]
    assert cb.main(moved) == 1 and "moved=True" in capsys.readouterr().out


@pytest.mark.parametrize("defect", ["untracked_src", "foreign_source", "no_blob", "meta_mismatch", "outside_tunes",
                                    "unreadable_meta", "missing_meta", "legacy_digest", "missing_target"])
def test_compile_bundle_seal_refuses(toy_repo, tmp_path, capsys, defect):
    cb = _load("compile_bundle")
    outside = tmp_path / "autotune"
    cache, blob, meta = _toy_cache(tmp_path, flags=f"--xla_gpu_per_fusion_autotune_cache_dir={outside}")
    source = toy_repo
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--repo", str(toy_repo), "--out", str(tmp_path / "b")]
    assert cb.main(seal + ["--source", str(source)]) == 0  # an EMPTY outside flag dir is only recorded
    if defect == "untracked_src":
        (toy_repo / "src/gpuwrf/extra.py").write_text("")
    elif defect == "foreign_source":
        (toy_repo / "src/gpuwrf/__init__.py").write_text('__version__ = "9.9.9"\n')
        _git(toy_repo, "add", "-A")
        _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "other")
        seal[seal.index("HEAD")] = "HEAD~1"
    elif defect == "no_blob":
        for p in [*(cache / "aot").rglob("*.xlaexec"), *(cache / "aot").rglob("*.meta")]:
            p.unlink()
    elif defect == "meta_mismatch":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.xlaexec").write_bytes(b"edited")
    elif defect == "unreadable_meta":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(b"not a pickle")
    elif defect == "missing_meta":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").unlink()
    elif defect == "legacy_digest":  # a digest-less record cannot attest which executable it describes
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, blob_sha256=None)))
    elif defect == "missing_target":
        fp = {k: v for k, v in meta.fingerprint.items() if k != "compute_capability"}
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, fingerprint=fp)))
    else:
        outside.mkdir()
        (outside / "t.textproto").write_text("tune escaped the seal")
    assert cb.main(seal + ["--source", str(source)]) == 1
    assert "SEAL_REFUSED" in capsys.readouterr().out


def test_wheel_smoke_run_refuses_without_wrf_tables_before_any_work(tmp_path):
    work, wrf, venv = tmp_path / "work", tmp_path / "WRF", tmp_path / "venv"
    (wrf / "run").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("#!/bin/sh\nexit 1\n")  # a stub: nothing can ever be installed anywhere
    (venv / "bin/python").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "GPUWRF_WRF_ROOT"}
    for extra in ([], ["--wrf-root", str(wrf)]):  # unset, and a tree without run/MPTABLE.TBL
        rc = subprocess.run(["bash", str(TOOLS / "wheel_smoke.sh"), "--venv", str(venv), "--work", str(work), "--run",
                             *extra], capture_output=True, text=True, env=env)
        assert rc.returncode == 2 and "MPTABLE.TBL" in rc.stderr and not work.exists()


def test_final_label_requires_the_approved_readme_sha(toy_repo, tmp_path):
    # toy repo + tmp lane: even without the guard nothing outside tmp_path is touched
    rc = subprocess.run(["bash", str(TOOLS / "final.sh"), "--version", "9.9.9", "--private", "HEAD", "--label", "final",
                         "--repo", str(toy_repo), "--base", "HEAD", "--lane", str(tmp_path / "lane")],
                        cwd=toy_repo, capture_output=True, text=True)
    assert rc.returncode == 2 and "--readme-sha256" in rc.stderr and not (tmp_path / "lane").exists()


def test_proofs_scope_rules():
    ps = _load("proofs_scope")
    mib = 1 << 20
    new = {"proofs/a/fixtures/x.npz": 9 * mib, "proofs/a/fixtures/y.npz": 1000, "proofs/a/fixtures/run.log": 10,
           "proofs/a/fixtures/archive.tar.gz": 10, "proofs/a/helper.py": 100, "proofs/a/data/z.npz": 100,
           "proofs/b/VERDICT.md": 100, "proofs/b/big.json": 2 * mib, "proofs/b/raw.npz": 100,
           "proofs/b/trace.nsys-rep": 10, "proofs/b/wrfout_d01_x": 10, "proofs/b/__pycache__/m.cpython-313.pyc": 10,
           "proofs/s2_integrate/b1/QUALIFICATION.json": 100, "proofs/s2_integrate/b1/integrate_b1.py": 100,
           "proofs/s2_integrate/b1/sanity.log.gz": 100, "proofs/t_redsuite/FINAL.json": 2 * mib,
           "proofs/c/huge.npz": 11 * mib}
    readers = {"tests/test_a.py": "ROOT/'proofs/a/fixtures'\nsys.path.insert(0, str(ROOT/'proofs/a'))",
               "tests/test_c.py": "np.load(ROOT / 'proofs/c/huge.npz')",
               "proofs/a/helper.py": "np.load('proofs/a/data/z.npz')", "docs/x.md": "see proofs/b/big.json"}
    c = ps.classify(new, readers)
    d = {p: c[p]["decision"] for p in c}
    assert d["proofs/a/fixtures/x.npz"] == d["proofs/a/fixtures/y.npz"] == "fixture"
    assert c["proofs/a/fixtures/x.npz"]["consumers"] == ["tests/test_a.py"]
    assert d["proofs/a/fixtures/run.log"] == d["proofs/a/fixtures/archive.tar.gz"] == "exclude"  # per-run log, archive
    assert d["proofs/a/helper.py"] == "fixture" and d["proofs/a/data/z.npz"] == "fixture"  # transitive via the helper
    assert d["proofs/b/VERDICT.md"] == "receipt" and d["proofs/b/big.json"] == "exclude"  # receipts <= 1 MiB only
    assert c["proofs/b/big.json"]["docs"] == ["docs/x.md"]
    assert d["proofs/b/raw.npz"] == d["proofs/b/trace.nsys-rep"] == d["proofs/b/wrfout_d01_x"] == "exclude"
    assert d["proofs/b/__pycache__/m.cpython-313.pyc"] == "refuse" and d["proofs/c/huge.npz"] == "refuse"
    assert d["proofs/s2_integrate/b1/QUALIFICATION.json"] == "receipt"
    assert d["proofs/s2_integrate/b1/integrate_b1.py"] == d["proofs/s2_integrate/b1/sanity.log.gz"] == "exclude"
    assert d["proofs/t_redsuite/FINAL.json"] == "exclude"
    assert ps.dependent_tests(c, readers) == {"tests/test_a.py": ["proofs/a/fixtures/archive.tar.gz",
                                                                  "proofs/a/fixtures/run.log"]}
    idx = ps.retained_index(c, {p: "s" for p in c})
    assert set(idx) == {p for p, v in d.items() if v == "exclude"} and idx["proofs/b/big.json"]["sha256"] == "s"


def test_release_readme_is_the_approved_versioned_readme(toy_repo, monkeypatch):
    monkeypatch.setattr(curate, "PRIV_REPO", str(toy_repo))
    (toy_repo / "docs/release").mkdir(parents=True)
    (toy_repo / "docs/release/README_v0.3_draft.md").write_text("# stale draft v0.3.4\n")
    (toy_repo / "README.md").write_text("# wrf_gpu v0.4.0\nForecasts on one GPU.\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "readme")
    sha = hashlib.sha256((toy_repo / "README.md").read_bytes()).hexdigest()
    text, rec = curate.release_readme("HEAD", "0.4.0", sha, 4000)
    assert text.startswith("# wrf_gpu v0.4.0") and rec["source"].endswith(":README.md") and rec["source_sha256"] == sha
    with pytest.raises(AssertionError, match="not the approved"):
        curate.release_readme("HEAD", "0.4.0", "0" * 64, 4000)
    with pytest.raises(AssertionError, match="never names v0.4.1"):
        curate.release_readme("HEAD", "0.4.1", None, 4000)
    with pytest.raises(AssertionError, match="budget"):
        curate.release_readme("HEAD", "0.4.0", None, 3)
    with pytest.raises(AssertionError, match="bytecode"):
        curate.refuse_bytecode(["src/a.py", "proofs/x/__pycache__/m.cpython-313.pyc"])
    curate.refuse_bytecode(["src/a.py", "proofs/x/m.py"])


def test_committed_scan_sees_force_added_files(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    base = _git(toy_repo, "rev-parse", "HEAD").strip()
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(dict(allowlist={"proofs/a/ok.json": {}}, exclusions={"proofs/a/run.log": {}})))
    (toy_repo / "proofs/a").mkdir(parents=True)
    (toy_repo / "proofs/a/ok.json").write_text("{}")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clean")
    clean = verify.committed(toy_repo, base, _git(toy_repo, "rev-parse", "HEAD").strip(), scope)
    assert clean["fails"] == [] and clean["new"] == 1
    (toy_repo / "proofs/a/__pycache__").mkdir()
    (toy_repo / "proofs/a/__pycache__/m.cpython-313.pyc").write_bytes(b"\x00<USER_HOME>/x")  # ignored, force-added
    (toy_repo / "proofs/a/run.log").write_text("log")
    (toy_repo / "proofs/a/extra.json").write_text("{}")
    (toy_repo / "tests/fixtures").mkdir(parents=True)
    (toy_repo / "tests/fixtures/tiny.cubin").write_bytes(b"\x7fELF")  # a test fixture, not a dump
    (toy_repo / "proofs/a/k.cubin").write_bytes(b"\x7fELF")
    (toy_repo / "env.txt").write_text("PATH=/usr/bin\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "add", "-f", "proofs/a/__pycache__/m.cpython-313.pyc")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "dirty")
    dirty = verify.committed(toy_repo, base, _git(toy_repo, "rev-parse", "HEAD").strip(), scope)
    assert dirty["bytecode"] == ["proofs/a/__pycache__/m.cpython-313.pyc"]
    assert "proofs/a/__pycache__/m.cpython-313.pyc" in dirty["private"]
    assert dirty["proofs_excluded_committed"] == ["proofs/a/run.log"]
    assert sorted(dirty["dumps"]) == ["env.txt", "proofs/a/k.cubin"]
    assert "proofs/a/extra.json" in dirty["proofs_not_allowlisted"]
    assert {"bytecode", "private", "proofs_excluded_committed", "proofs_not_allowlisted"} <= set(dirty["fails"])
    assert "scope_missing" in verify.committed(toy_repo, base, "HEAD", None)["fails"]


OPS_TEXT = ("her" + "dr lane notes\n").encode()  # an ops-marker word, assembled so this public test never carries it


def test_proofs_scope_round_trip_and_test_refs(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    monkeypatch.setattr(curate, "PRIV_REPO", str(toy_repo))
    base = _git(toy_repo, "rev-parse", "HEAD").strip()
    files = {"proofs/a/fixtures/f.npz": b"F", "proofs/a/fixtures/run.log": b"L", "proofs/a/VERDICT.md": b"# v\n",
             "proofs/a/raw.npz": b"R", "tests/test_a.py": b"DATA = ROOT/'proofs/a/fixtures'\n",
             "proofs/a/OPS.md": OPS_TEXT}
    for rel, data in files.items():
        (toy_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (toy_repo / rel).write_bytes(data)
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "proofs")
    priv = _git(toy_repo, "rev-parse", "HEAD").strip()
    out, work = tmp_path / "out", tmp_path / "work"
    shutil.copytree(toy_repo, out, ignore=shutil.ignore_patterns(".git"))
    work.mkdir()
    base_set = set(_git(toy_repo, "ls-tree", "-r", "--name-only", base).split())
    (out / "proofs/a/OPS.md").unlink()  # publish_all never copies ops-marker files
    ops = {"proofs/a/OPS.md": (len(OPS_TEXT), hashlib.sha256(OPS_TEXT).hexdigest())}
    rep = curate.apply_proofs_scope(out, base_set, priv, "0.4.0", work, ops)
    assert json.loads((out / "proofs/RETAINED_PRIVATELY.json").read_text())["files"]["proofs/a/OPS.md"]["sha256"] == ops[
        "proofs/a/OPS.md"][1]
    assert rep["exclude_files"] == 2 and not (out / "proofs/a/raw.npz").exists() and (out / "proofs/a/fixtures/f.npz").exists()
    scope = json.loads((work / "proofs_scope.json").read_text())
    assert scope["dependent_tests"] == {"tests/test_a.py": ["proofs/a/fixtures/run.log"]}
    assert scope["allowlist"]["proofs/a/fixtures/f.npz"]["consumers"] == ["tests/test_a.py"]
    good = verify.proofs_scope(out, priv, base)
    assert good["index_exact"] and not good["missing_published"] and not good["excluded_present"]
    assert good["ops_private"] == 1
    assert verify.test_refs(out, priv)["missing"] == []
    (out / "proofs/a/raw.npz").write_bytes(b"R")  # an excluded file reappears
    assert verify.proofs_scope(out, priv, base)["excluded_present"] == ["proofs/a/raw.npz"]
    (out / "proofs/a/raw.npz").unlink()
    (out / "proofs/a/fixtures/f.npz").unlink()  # a consumed fixture goes missing
    assert verify.proofs_scope(out, priv, base)["missing_published"] == ["proofs/a/fixtures/f.npz"]
    assert verify.test_refs(out, priv)["missing"] == [("tests/test_a.py", "proofs/a/fixtures")]
    (out / "proofs/a/fixtures/f.npz").write_bytes(b"F")
    index = json.loads((out / "proofs/RETAINED_PRIVATELY.json").read_text())
    index["files"].pop("proofs/a/raw.npz")
    (out / "proofs/RETAINED_PRIVATELY.json").write_text(json.dumps(index))
    assert verify.proofs_scope(out, priv, base)["index_exact"] is False


def test_verify_binds_the_published_readme_bytes(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    readme = b"# wrf_gpu v0.4.0\nForecasts on one GPU.\n"
    (toy_repo / "README.md").write_bytes(readme)
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "readme")
    approved = hashlib.sha256(readme).hexdigest()
    pub = tmp_path / "pub"
    pub.mkdir()
    (pub / "README.md").write_bytes(readme)
    good = verify.readme_check(pub, "HEAD", "HEAD~1", "0.4.0", None, approved)
    assert good["approved_match"] is True and good["published_is_source"] and good["published_sha256"] == approved
    (pub / "README.md").write_bytes(readme + b"Also ships an unapproved claim.\n")  # public-only tamper, source unchanged
    bad = verify.readme_check(pub, "HEAD", "HEAD~1", "0.4.0", None, approved)
    assert bad["source_sha256"] == approved and bad["approved_match"] is False and not bad["published_is_source"]
