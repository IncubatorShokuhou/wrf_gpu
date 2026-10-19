#!/usr/bin/env bash
# Starts cpu_ref.sh only after w41-rw3add posts its exact 'RW3ADD GPU END' line (manager 00:06:31Z, QUIET rule):
# anchored on the sender at line start, only lines appended after arming, /usr/bin/grep (V042 lesson), then waits
# until /tmp/wrf_gpu2_quiet is gone.  usage: setsid bash -c 'echo $$ > PIDFILE; exec bash wait_rw3add_end.sh'
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0
N=$(wc -l < "$MBX")
echo "$(date -u +%FT%TZ) armed at line $N" >> "$L/wait_rw3add.log"
until tail -n +"$((N + 1))" "$MBX" | /usr/bin/grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}(\.[0-9]+)?(Z|\+00:00) w41-rw3add: RW3ADD GPU END'; do
  sleep 60
done
echo "$(date -u +%FT%TZ) END line seen: $(tail -n +"$((N + 1))" "$MBX" | /usr/bin/grep -m1 -E '^.{20,40} w41-rw3add: RW3ADD GPU END' | cut -c1-160)" >> "$L/wait_rw3add.log"
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
echo "$(date -u +%FT%TZ) quiet clear -> cpu_ref.sh" >> "$L/wait_rw3add.log"
CORES=8,9,12 exec bash "$HERE/cpu_ref.sh" >> "$L/wait_rw3add.log" 2>&1
