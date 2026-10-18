#!/usr/bin/env bash
# Consumer pytest files one at a time, each capped at 1 GiB (no RAM slot needed). usage: consumers.sh TREE FILELIST OUTDIR
TREE=$1; LIST=$2; OUT=$3; mkdir -p "$OUT"; cd "$TREE"
while read -r f; do
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  tag=$(echo "$f" | tr '/' '_')
  [ -e "$OUT/$tag.log" ] && continue
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 timeout 900 systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -- \
    taskset -c 8,9,12,13 nice -n 19 python -m pytest -q -p no:cacheprovider -x --basetemp <USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/pytest/$tag "$f" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "$rc $f $(tail -1 "$OUT/$tag.log" | cut -c1-150)" >> "$OUT/summary.txt"
done < "$LIST"
echo DONE >> "$OUT/summary.txt"
