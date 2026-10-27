#!/usr/bin/env bash
# Print real UTC epoch = system epoch corrected by chrony's measured offset (system clock +~1 h after the 2026-10-20 wake-up, slewing).
# Usage: real_utc.sh            -> epoch seconds (integer)
#        real_utc.sh --iso      -> ISO UTC
o=$(chronyc tracking 2>/dev/null | awk '/^System time/{v=$4; if($6=="fast") print -v; else print v}')
[ -n "$o" ] || o=0
e=$(python3 -c "import time;print(int(time.time()+($o)))")
if [ "${1:-}" = --iso ]; then date -u -d "@$e" +%FT%TZ; else echo "$e"; fi
