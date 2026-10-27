#!/usr/bin/env bash
# Per-phase hard memory cap; independent of the shared RAM-slot FIFO.
# usage: with_memory.sh [--max SIZE] -- cmd...   (default 12G; the manifest sets the GPU-phase size)
set -euo pipefail
MAX=12G
if [[ ${1:-} == --max ]]; then [[ ${2:-} =~ ^[0-9]+G$ ]] || exit 2; MAX=$2; shift 2; fi
[[ ${1:-} == -- ]] || exit 2
shift
[[ $# -gt 0 ]] || exit 2
# Preserve Bash command text literally through systemd (including $ expansions).
exec systemd-run --user --scope --quiet --expand-environment=no \
  -p MemoryMax="$MAX" -p MemorySwapMax=0 -- "$@"
