#!/usr/bin/env bash
# w44-rc: (re)freeze tools_v044. Writes SHA256SUMS over every tool file the entries execute or read (all regular files
# except the 4 entry files, SHA256SUMS itself, prepare outputs and freeze.sh), then writes that SHA into the 4 entry files
# (TOOLS_SUMS_SHA=...). Any change after a LAUNCH SMOKE invalidates it: re-freeze, re-smoke, re-post the launcher SHAs.
set -euo pipefail
cd <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5
ENTRIES=(entry_prime_swiss.sh entry_prime_prod.sh entry_prime_wn3.sh entry_rw_warm.sh entry_rw3.sh entry_rw_short.sh cuda_mini_short.sh cuda_mini_rw3.sh)
rm -f SHA256SUMS
find . -maxdepth 1 -type f ! -name SHA256SUMS ! -name freeze.sh ! -name PIN_MANIFEST.json ! -name PREREG.json \
  ! -name autotune_seed_v043.sha256 ! -name prepare.log $(printf '! -name %s ' "${ENTRIES[@]}") -printf '%f\n' | sort \
  | xargs sha256sum > SHA256SUMS
S=$(sha256sum SHA256SUMS | cut -d' ' -f1)
for e in "${ENTRIES[@]}"; do sed -i "s/^TOOLS_SUMS_SHA=.*/TOOLS_SUMS_SHA=$S/" "$e"; chmod 0555 "$e"; done
chmod 0444 SHA256SUMS
echo "FROZEN SHA256SUMS $S ($(wc -l < SHA256SUMS) files)"
sha256sum "${ENTRIES[@]}"
