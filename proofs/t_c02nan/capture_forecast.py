"""Host-only snapshots at existing c02 history boundaries; no step instrumentation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot-steps", default="2400,2600")
    args = parser.parse_args()
    source = args.source.resolve() if args.source else Path(__file__).resolve().parents[2]
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    fd = int(os.environ["GPUWRF_GPU_LOCK_FD"])
    assert fd >= 3
    assert os.fstat(fd).st_ino == os.stat(os.environ["GPUWRF_GPU_LOCK_FILE"]).st_ino
    assert subprocess.run(["git", "-C", str(source), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 0
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    args.arm.mkdir(parents=True, exist_ok=True)
    assert not (args.arm / "receipt.json").exists()
    import gpuwrf  # noqa: F401
    import jax
    import numpy as np
    assert jax.devices()[0].platform == "gpu", jax.devices()
    print(f"TCN platform=gpu source={source} device={jax.devices()[0]}", flush=True)
    stop = threading.Event()
    memory_samples = []
    def sample_memory():
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                for line in result.stdout.splitlines():
                    fields = line.split(",")
                    if len(fields) == 2 and fields[0].strip() == str(os.getpid()):
                        memory_samples.append(int(fields[1].strip()))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.operational_mode import build_clock_base
    from gpuwrf.diagnostics.census import write_segment

    snapshot_steps = {int(s) for s in args.snapshot_steps.split(",") if s}
    original = pipeline._PerDomainWrfoutWriter.__call__

    def capture(self, name, own_step, carry):
        if int(own_step) in snapshot_steps:
            bundle = self.bundles[name]
            payload = jax.device_get(dict(carry=carry, namelist=bundle.namelist,
                clock_base=build_clock_base(bundle.namelist), step=int(own_step)))
            path = args.arm / f"{name}_step{own_step}.pkl"
            with path.open("xb") as stream:
                pickle.dump(payload, stream, protocol=5)
            stats = {}
            for field in ("u", "v", "w", "theta", "p", "ph", "mu", "qv", "qc", "qr", "qi", "qs"):
                value = np.asarray(getattr(payload["carry"].state, field))
                finite = np.isfinite(value)
                stats[field] = dict(nonfinite=int(value.size-finite.sum()),
                    first_bad=None if finite.all() else list(map(int, np.argwhere(~finite)[0])),
                    min=float(value[finite].min()) if finite.any() else None,
                    max=float(value[finite].max()) if finite.any() else None)
            (args.arm / f"{name}_step{own_step}.json").write_text(json.dumps(stats, indent=2)+"\n")
            census_dir = args.arm / f"census_step{own_step}"
            census_dir.mkdir()
            write_segment(census_dir, {name: carry}, {name: bundle},
                {name: int(own_step)}, self, self.output_cadence_steps)
            print(f"SNAPSHOT step={own_step} path={path} bytes={path.stat().st_size} stats={stats}", flush=True)
        return original(self, name, own_step, carry)

    pipeline._PerDomainWrfoutWriter.__call__ = capture
    argv = ["run", "--input-dir", str(args.input), "--output-dir", str(args.arm / "gpu"),
        "--proof-dir", str(args.arm / "proof"), "--scratch-dir", str(args.arm / "scratch"),
        "--hours", str(args.hours), "--domains-from-namelist"]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    rc = 1
    try:
        from gpuwrf.cli import main as cli_main
        rc = cli_main(argv)
    finally:
        stop.set()
        sampler.join(timeout=6)
        receipt = dict(platform=jax.devices()[0].platform, source=str(source),
            source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_s=time.monotonic()-t0, rc=rc, argv=argv,
            scope="diagnostic host snapshots at existing history boundaries; not a timing benchmark",
            cache=os.environ.get("GPUWRF_JAX_CACHE_DIR"),
            sampled_process_peak_vram_MiB=max(memory_samples, default=None),
            memory_scope="once/sec process samples; short peaks can be missed",
            input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.input.iterdir() if p.is_file()},
            environment={k: os.environ.get(k) for k in ("GPUWRF_FAST_DEFAULTS", "GPUWRF_CENSUS",
                "GPUWRF_CENSUS_SLIM", "GPUWRF_RRTMG_MP_RE", "XLA_CLIENT_MEM_FRACTION")})
        (args.arm / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
