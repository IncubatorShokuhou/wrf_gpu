#!/usr/bin/env bash
# Exact-call sequencer; SEQ_SMOKE=1 exercises all phases on CPU/private storage.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${GATE_PYTHON:-python3}" "$here/gpu_gate.py" sequence "$@"
