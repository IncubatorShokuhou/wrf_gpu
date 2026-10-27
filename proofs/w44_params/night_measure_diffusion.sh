#!/usr/bin/env bash
# V3 (launcher v3: first-entry PASS gate, E225 guard, own scopes; v2 withdrawn): prepared-cache/E41 gate, then actual product-schedule warm A/B.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec bash "$HERE/launch_step.sh" diffusion warm "$@"
