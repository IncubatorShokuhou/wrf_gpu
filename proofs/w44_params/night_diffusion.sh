#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PARAMS_FAMILY=diffusion
exec bash "$HERE/night_launch.sh" "$@"
