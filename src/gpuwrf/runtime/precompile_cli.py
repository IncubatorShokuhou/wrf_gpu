"""Queue ordinary GPU cache priming for a list of complete WRF namelists.

CPU CompileOnly artifacts remain qualification-only until the W4 GPU byte gate
passes. This command deliberately uses the existing forecast entry point.
Imports here must not initialize a JAX backend (the CLI configures C-auto first).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA = "gpuwrf-precompile-v1"


def register(subparsers):
    p = subparsers.add_parser("precompile", help="queue GPU cache priming for namelist variants")
    p.add_argument("--namelist-variants", required=True, type=Path, help="JSON {variants: [{name, namelist}]}")
    p.add_argument("--input-dir", required=True, type=Path, help="native WRF input directory shared by the variants")
    p.add_argument("--cache-dir", required=True, type=Path, help="writable shared JAX/AOT cache (not a sealed bundle)")
    p.add_argument("--work-dir", required=True, type=Path, help="disk-backed logs, receipts and temporary forecast outputs")
    p.add_argument("--hours", type=int, default=3, help="priming forecast length; must align with every root timestep")
    p.add_argument("--timeout-seconds", type=int, default=1800, help="timeout per variant, including initialization/compilation")
    p.add_argument("--execute", action="store_true", help="execute the queue in an authorized GPU window under with_gpu_lock.sh")
    p.set_defaults(func=command)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def variants(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"variants"} or not isinstance(data["variants"], list) or not data["variants"]:
        raise ValueError("expected a nonempty JSON object {variants: [{name, namelist}]}")
    result, names = [], set()
    for row in data["variants"]:
        if not isinstance(row, dict) or set(row) != {"name", "namelist"}:
            raise ValueError("each variant requires exactly name and namelist")
        name = row["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in names:
            raise ValueError("variant names must be unique simple names (letters, digits, '_' and '-')")
        if not isinstance(row["namelist"], str) or not row["namelist"]:
            raise ValueError(f"{name}: namelist must be a path")
        nml = (path.parent / row["namelist"]).resolve()
        if not nml.is_file():
            raise ValueError(f"{name}: missing namelist {nml}")
        names.add(name)
        result.append({"name": name, "namelist": str(nml), "namelist_sha256": sha_file(nml)})
    return result


def validate_variant(row, hours):
    from gpuwrf.io.gen2_accessor import parse_namelist
    from gpuwrf.io.namelist_check import validate_operational_namelist
    nml = Path(row["namelist"])
    validate_operational_namelist(nml)
    config = parse_namelist(nml)
    domains = config.get("domains", {})
    count = domains.get("max_dom", 1)
    count = count[0] if isinstance(count, list) else count
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{row['name']}: max_dom must be positive")
    dt = domains.get("time_step", 0)
    dt = dt[0] if isinstance(dt, list) else dt
    num = domains.get("time_step_fract_num", 0)
    den = domains.get("time_step_fract_den", 1)
    num = num[0] if isinstance(num, list) else num
    den = den[0] if isinstance(den, list) else den
    from fractions import Fraction
    try:
        timestep = Fraction(str(dt)) + Fraction(str(num)) / Fraction(str(den))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{row['name']}: invalid root timestep") from exc
    if timestep <= 0 or (Fraction(hours * 3600) / timestep).denominator != 1:
        raise ValueError(f"{row['name']}: --hours must contain an integer number of root timesteps")
    return count


def source_identity():
    package = Path(__file__).resolve().parents[1]
    # Include packaged data as well as code: changing a baked table invalidates a
    # receipt even when the package version and source code have not changed.
    files = {str(p.relative_to(package)): sha_file(p) for p in sorted(package.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    from gpuwrf import assets
    # Repository tables may live outside the package. Missing remote assets
    # retain their pin; a later verified download has exactly the same digest.
    tables = {}
    for rel, info in assets.manifest()["assets"].items():
        path = assets.asset_path(rel)
        tables[rel] = sha_file(path) if path.is_file() else info["sha256"]
    versions = {name: metadata.version(name) for name in ("jax", "jaxlib", "numpy")}
    return {"package_sha256": digest(files), "tables_sha256": digest(tables),
            "versions": versions, "python": sys.version}


def cache_files(cache):
    return {str(p.relative_to(cache)): sha_file(p) for p in sorted(cache.rglob("*"))
            if p.is_file() and (p.name.endswith(("-cache", ".xlaexec", ".meta")) or
                                "xla_gpu_per_fusion_autotune_cache_dir" in p.parts)}


def input_files(input_dir):
    files, visited = {}, set()
    for root, dirs, names in os.walk(input_dir, followlinks=True):
        resolved = Path(root).resolve()
        if resolved in visited:
            raise ValueError("input directory has cyclic or repeated directory symlinks")
        visited.add(resolved)
        dirs[:] = [d for d in dirs if not d.startswith(("wrfout", "rsl."))]
        for name in sorted(names):
            rel = (Path(root) / name).relative_to(input_dir)
            if rel.as_posix() != "namelist.input" and not name.startswith(("wrfout", "rsl.")):
                files[rel.as_posix()] = sha_file(input_dir / rel)
    return files


def cached(entry, identity, cache):
    if not isinstance(entry, dict) or entry.get("identity") != identity or entry.get("status") != "BUILT":
        return False
    files = entry.get("files", {})
    if not files or not any(name.endswith(".xlaexec") for name in files):
        return False
    return all((cache / name).is_relative_to(cache) and ".." not in Path(name).parts
               and (cache / name).is_file() and sha_file(cache / name) == sha for name, sha in files.items())


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def lock_fd():
    from gpuwrf.runtime.gpu_preflight import gpu_lock_status
    status = gpu_lock_status()
    if not status["ok"]:
        raise ValueError(f"--execute requires scripts/with_gpu_lock.sh: {status['reason']}")
    fd = int(status["fd"])
    official = Path("/tmp/wrf_gpu2_gpu.lock").stat()
    held = os.fstat(fd)
    if fd <= 2 or (held.st_dev, held.st_ino) != (official.st_dev, official.st_ino):
        raise ValueError("GPU lock descriptor does not name the official lock")
    return fd


def target_identity():
    # Called only under the official GPU lock. No JAX backend or CUDA pool in
    # the parent: each priming child starts with the GPU free of a parent pool.
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version,name,compute_cap",
                                     "--format=csv,noheader"], text=True, timeout=15)
    return {"nvidia_smi": query.strip(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


@contextmanager
def case_view(input_dir, namelist, directory):
    """The driver reads input_dir/namelist.input; bind the selected variant there."""
    view = directory / "input"
    view.mkdir()
    # Freeze the chosen namelist, and keep meteorological inputs read-only.
    shutil.copyfile(namelist, view / "namelist.input")
    for path in input_dir.iterdir():
        if path.name != "namelist.input" and not path.name.startswith(("wrfout", "rsl.")):
            (view / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    yield view


def run_worker(request, log, timeout, fd, env):
    argv = [sys.executable, "-m", "gpuwrf.runtime.precompile_cli", "--worker-request", str(request)]
    with log.open("w") as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, env=env,
                                 pass_fds=(fd,), start_new_session=True)
        def interrupted(signum, frame):
            raise InterruptedError("precompile interrupted")
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            return child.wait(timeout=timeout)
        except BaseException:
            # A timed-out compile must release every descendant before the outer
            # official GPU lock is released (never leave an orphan GPU process).
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            # The immediate child may exit before a descendant that ignored
            # TERM. Kill its remaining process group before returning the lock.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def worker(request_path):
    """Fresh child; the public CLI configures its allocator before any devices()."""
    request = json.loads(request_path.read_text())
    lock_fd()
    from gpuwrf.cli import main
    if source_identity() != request["source"] or input_files(Path(request["argv"][2])) != request["inputs"]:
        raise ValueError("source or inputs changed since precompile admission")
    started = time.monotonic()
    rc = main(request["argv"])
    import jax
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise ValueError("precompile priming requires the GPU backend")
    from gpuwrf.runtime.aot_executable import target_fingerprint
    from gpuwrf.runtime.domain_tree import NESTED_AOT_STATUS
    from gpuwrf.runtime.aot_precompile import _aot_blob_paths
    programs = {}
    if rc == 0:
        import pickle
        for name, status in (NESTED_AOT_STATUS.get("domains") or {}).items():
            found = False
            for address in ("cheap_key", "hlo_sha256"):
                key = status.get(address)
                paths = _aot_blob_paths(name, **{address: key}) if key else None
                if not paths or not all(p.is_file() for p in paths):
                    continue
                with paths[1].open("rb") as f:
                    meta = pickle.load(f)
                if meta.blob_sha256 != sha_file(paths[0]):
                    continue
                programs[name] = {"address": address, "key": key, "blob": str(paths[0]),
                                  "meta": str(paths[1]), "blob_sha256": meta.blob_sha256}
                found = True
                break
            if not found:
                raise ValueError(f"{name}: priming did not persist this program's native and metadata")
        if not programs:
            raise ValueError("priming produced no AOT program inventory")
    _write(Path(request["result"]), {"rc": rc, "platform": device.platform,
           "fingerprint": target_fingerprint(device), "wall_s": time.monotonic() - started,
           "aot_programs": programs})
    return rc


def command(args):
    try:
        return _command(args)
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"gpuwrf precompile: {exc}", file=sys.stderr)
        return 2


def _command(args):
    if args.hours <= 0 or args.timeout_seconds <= 0:
        raise ValueError("hours and timeout-seconds must be positive")
    fd = lock_fd() if args.execute else None
    if args.execute and os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
        raise ValueError("--execute requires JAX_PLATFORMS=gpu (CPU deviceless is not qualified)")
    input_dir, cache, work = (p.expanduser().resolve() for p in (args.input_dir, args.cache_dir, args.work_dir))
    from gpuwrf.cli import _is_tmpfs
    if _is_tmpfs(work) or _is_tmpfs(cache):
        raise ValueError("cache-dir and work-dir must be disk-backed, not tmpfs")
    if not input_dir.is_dir():
        raise ValueError(f"missing input directory: {input_dir}")
    if work == input_dir or work.is_relative_to(input_dir) or input_dir.is_relative_to(work):
        raise ValueError("work-dir and input-dir must be separate directories")
    if cache == work or cache.is_relative_to(work) or work.is_relative_to(cache):
        raise ValueError("cache-dir and work-dir must be separate directories")
    if cache == input_dir or cache.is_relative_to(input_dir) or input_dir.is_relative_to(cache):
        raise ValueError("cache-dir and input-dir must be separate directories")
    from gpuwrf.runtime.aot_key_cli import sealed_bundle_of
    if sealed_bundle_of(cache):
        raise ValueError("use a writable cache overlay; precompile must not write into a sealed bundle")
    rows = variants(args.namelist_variants.expanduser().resolve())
    counts = {row["name"]: validate_variant(row, args.hours) for row in rows}
    for domain in range(1, max(counts.values()) + 1):
        p = input_dir / f"wrfinput_d{domain:02d}"
        if not p.is_file():
            raise ValueError(f"missing native input {p}")
    if not (input_dir / "wrfbdy_d01").is_file():
        raise ValueError("missing native input wrfbdy_d01")
    # Include auxiliary input streams and tables too. Ignore CPU history files:
    # all variants use the native initialization entry point.
    inputs = input_files(input_dir)
    context = {"schema": SCHEMA, "source": source_identity(), "inputs": inputs, "hours": args.hours,
               "cache": str(cache), "environment_sha256": digest({k: v for k, v in os.environ.items()
               if k.startswith(("GPUWRF_", "JAX_", "XLA_", "CUDA_")) and not
               k.startswith("GPUWRF_GPU_LOCK_")})}
    if args.execute:
        context["target"] = target_identity()
    index_path = work / "index.json"
    report = {"schema": SCHEMA, "backend": "queued-gpu", "deviceless": "NOT_QUALIFIED",
              "execute": args.execute, "input_dir": str(input_dir), "cache_dir": str(cache),
              "work_dir": str(work), "hours": args.hours, "variants": [],
              "execute_argv": [sys.executable, "-m", "gpuwrf.cli", "precompile", "--namelist-variants",
                  str(args.namelist_variants.resolve()), "--input-dir", str(input_dir), "--cache-dir", str(cache),
                  "--work-dir", str(work), "--hours", str(args.hours),
                  "--timeout-seconds", str(args.timeout_seconds), "--execute"]}
    work.mkdir(parents=True, exist_ok=True)
    # Serialize this tool's receipt updates across processes sharing a work dir.
    import fcntl
    with (work / "precompile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        for row in rows:
            name = row["name"]
            identity = digest({"context": context, "namelist_sha256": row["namelist_sha256"]})
            item = {**row, "identity": identity, "max_dom": counts[name]}
            if args.execute and cached(index.get(identity), identity, cache):
                item["status"] = "CACHED"
            elif not args.execute:
                item["status"] = "QUEUED"
            else:
                cache.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, GPUWRF_JAX_CACHE="1", GPUWRF_JAX_CACHE_DIR=str(cache),
                           JAX_COMPILATION_CACHE_DIR=str(cache), JAX_PLATFORMS="gpu")
                with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=work) as temp:
                    attempt = Path(temp)
                    with case_view(input_dir, Path(row["namelist"]), attempt) as view:
                        if sha_file(view / "namelist.input") != row["namelist_sha256"]:
                            raise ValueError(f"{name}: namelist changed during admission")
                        request = {"argv": ["run", "--input-dir", str(view), "--domains-from-namelist",
                                   "--hours", str(args.hours), "--output-dir", str(attempt / "output"),
                                   "--scratch-dir", str(attempt / "scratch")], "result": str(attempt / "result.json"),
                                   "source": context["source"], "inputs": inputs}
                        request_path = attempt / "request.json"
                        _write(request_path, request)
                        log = work / f"{name}-{identity[:12]}.log"
                        try:
                            rc = run_worker(request_path, log, args.timeout_seconds, fd, env)
                        except subprocess.TimeoutExpired:
                            rc = 124
                        result_path = attempt / "result.json"
                        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                        if rc or result.get("rc") != 0 or result.get("platform") != "gpu" or not result.get("aot_programs"):
                            item.update(status="FAILED", returncode=rc, result=result, log=str(log))
                        elif source_identity() != context["source"] or input_files(input_dir) != inputs:
                            item.update(status="FAILED", reason="source or inputs changed during priming", log=str(log))
                        else:
                            files = cache_files(cache)
                            if not any(f.endswith(".xlaexec") for f in files):
                                raise ValueError(f"{name}: run produced no AOT executable; not marking cached")
                            item.update(status="BUILT", result=result, log=str(log))
                            index[identity] = {"identity": identity, "status": "BUILT", "files": files,
                                               "context": context, "result": result}
                            _write(index_path, index)
            report["variants"].append(item)
            _write(work / "report.json", report)
    print(json.dumps(report, indent=2))
    return 1 if any(v["status"] == "FAILED" for v in report["variants"]) else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker-request", required=True, type=Path)
    raise SystemExit(worker(p.parse_args().worker_request))
