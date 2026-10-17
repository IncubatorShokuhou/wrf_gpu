"""Run genuine source deletions in private overlays, never edit a measured tree."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mutants = [
        ("M1-no-wind-perturbation", "perturb/inputs.py", "changed = value * np.float32(factor)",
         "changed = value * np.float32(1)", "tests/test_perturb.py::test_case_copies_coherent_wind_and_preserves_mass_metadata_sources"),
        ("M2-standard-complex-rotation", "physics/stochastic.py", "real = amp * (real * vertical_cosine - imag * vertical_sine)",
         "original_real = real\n        real = amp * (real * vertical_cosine - imag * vertical_sine)",
         "tests/test_stochastic_primitives.py::test_original_fftpack_spectral_synthesis"),
        ("M3-missing-stochastic-binding", "io/namelist_binding.py",
         'changed["stochastic_config"] = stochastic',
         'changed["stochastic_config"] = None',
         "tests/test_stochastic_binding.py::test_independent_stochastic_option_reaches_central_pipeline_binding"),
    ]
    rows = []
    for name, filename, before, after, selection in mutants:
        directory = args.output / name
        source = directory / "src"
        shutil.copytree(args.tree / "src/gpuwrf", source / "gpuwrf", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Eager physics imports locate the tracked BASE Thompson tables relative
        # to the overlay root too. Link those immutable inputs, never copy truth.
        (directory/"data").symlink_to((args.tree/"data").resolve(),target_is_directory=True)
        path = source / "gpuwrf" / filename
        text = path.read_text()
        if before:
            assert text.count(before) == 1
            text = text.replace(before, after)
        if name.startswith("M2"):
            text = text.replace("imag = amp * (real * vertical_sine + imag * vertical_cosine)",
                                "imag = amp * (original_real * vertical_sine + imag * vertical_cosine)")
        path.write_text(text)
        command = [sys.executable, "-m", "pytest", "-q", selection, "-o", f"pythonpath={source}",
                   "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/u2-stoch/pytest"]
        result = subprocess.run(command, cwd=args.tree,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONPATH": str(source)},
            text=True, capture_output=True, timeout=180)
        log = result.stdout + result.stderr
        (directory / "pytest.log").write_text(log)
        killed = result.returncode == 1 and "AssertionError" in log and "ERROR " not in log
        rows.append({"mutant": name, "returncode": result.returncode, "genuine_assertion_kill": killed,
                     "selection": selection, "log": str(directory / "pytest.log")})
        # Keep the exact changed file and evidence; discard copied Python tree.
        shutil.copyfile(path, directory / Path(filename).name)
        shutil.rmtree(source)
        (directory/"data").unlink()
    (args.output / "receipt.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    assert all(row["genuine_assertion_kill"] for row in rows), rows


if __name__ == "__main__":
    main()
