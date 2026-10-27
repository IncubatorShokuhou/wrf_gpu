#!/usr/bin/env bash
# GA_START_UTC/GA_END_UTC only; cold GPU work <=15 min, E41 CPU after unlock.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python "$HERE/outer_supervise.py" --seconds 2100 -- bash "$HERE/night_launch.sh" prime
