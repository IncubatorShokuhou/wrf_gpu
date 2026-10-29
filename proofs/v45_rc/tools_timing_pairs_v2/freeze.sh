#!/usr/bin/env bash
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
cd <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2
ENTRIES=(entry_prime_swiss.sh entry_prime_prod.sh entry_prime_wn3.sh entry_rw_warm.sh entry_rw3.sh entry_rw_short.sh cuda_mini_short.sh cuda_mini_rw3.sh)
find . -type f ! -path '*/__pycache__/*' ! -name SHA256SUMS ! -name PIN_MANIFEST.json ! -name PREREG.json ! -name NIGHTLY_DONE.json ! -name prepare.log ! -name '*.pyc' $(printf '! -name %s ' "${ENTRIES[@]}") -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS
S=$(sha256sum SHA256SUMS | cut -d' ' -f1)
for e in "${ENTRIES[@]}"; do sed -i "s/^TOOLS_SUMS_SHA=.*/TOOLS_SUMS_SHA=$S/" "$e"; done
printf 'FROZEN %s\n' "$S"
sha256sum "${ENTRIES[@]}"
