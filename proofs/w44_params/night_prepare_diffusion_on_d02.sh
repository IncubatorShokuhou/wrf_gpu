#!/usr/bin/env bash
# V3 (launcher v3: first-entry PASS gate, E225 guard, own scopes; v2 withdrawn): one measured, resumable domain/value preparation; GA windows only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec bash "$HERE/launch_step.sh" diffusion prepare on d02 "$@"
