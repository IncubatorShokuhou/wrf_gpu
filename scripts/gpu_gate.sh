#!/usr/bin/env bash
# Configured admission gate. See docs/gpu_gate.md; no operational paths implicit.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${GATE_PYTHON:-python3}" "$here/gpu_gate.py" gate "$@"
