#!/usr/bin/env bash
# w42-bundle042: QUIET pause of a WHOLE session (all process groups of session id SID), complement to followup_quiet.sh:
# `timeout` (and anything else calling setpgid) moves the follow-up's E41 / D6 / icloud0 steps into their own process
# group, so a STOP/CONT of the follow-up's leader group misses them (found 2026-10-19 15:59Z: E41 ran on during HERO W0).
# Every second: QUIET present -> SIGSTOP every process group of SID; absent -> SIGCONT them. Exits when SID is empty.
# usage: setsid bash quiet_pause_sid.sh SID      (SID = the follow-up session leader pid, $L/followup.pgid)
set -u
SID=$1; Q=${FQ_QUIET_FILE:-/tmp/wrf_gpu2_quiet}; L=${FQ_LOGDIR:-<USER_HOME>/wrf_gpu2_lanes/w42-bundle042}
log() { echo "$(date -u +%FT%TZ) $*" >> "$L/quiet_pause_sid.log"; }
[[ $SID =~ ^[0-9]+$ ]] && (( SID > 1 )) || { log "bad SID '$SID'"; exit 2; }
groups() { ps -o pgid= -s "$SID" 2>/dev/null | tr -d ' ' | sort -u | /usr/bin/grep -E '^[0-9]+$'; }
log "armed SID $SID groups $(groups | tr '\n' ' ')"
state=
while [ -n "$(groups)" ]; do
  if [ -e "$Q" ]; then want=STOP; else want=CONT; fi
  if [ "$want" != "$state" ]; then
    for g in $(groups); do kill -"$want" -- "-$g" 2>/dev/null; done
    state=$want; log "$want groups $(groups | tr '\n' ' ')${want/CONT/}$( [ "$want" = STOP ] && echo " ($(cat "$Q" 2>/dev/null))")"
  elif [ "$want" = STOP ]; then   # a group created while stopped (new step) gets stopped too
    for g in $(groups); do kill -STOP -- "-$g" 2>/dev/null; done
  fi
  sleep 1
done
log "SID $SID gone"
