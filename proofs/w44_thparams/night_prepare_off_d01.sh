#!/usr/bin/env bash
# Manager chooses the window; same shared cache for every preparation arm.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python "$HERE/outer_supervise.py" --seconds 930 -- bash "$HERE/night_prepare.sh" off d01
