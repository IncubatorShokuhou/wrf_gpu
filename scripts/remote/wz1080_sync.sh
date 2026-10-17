#!/usr/bin/env bash
# wz1080_sync.sh — ship a local commit to the wz1080 test box and check it out DETACHED. No credentials:
# a git bundle travels over the ssh key (real commit SHAs; first sync = full history of the commit ≈ 1.7 GB,
# later syncs = incremental bundles against every commit the box already has).
#
# Usage: scripts/remote/wz1080_sync.sh <commit-ish> [--main] [--branch NAME] [--keep N] [--no-checkout]
#   default      immutable snapshot ~/src/wrf_gpu2_wt/<sha12> on the box (AGENTS: measure snapshots only)
#   --main       move the box's main clone ~/src/wrf_gpu2 instead (refused if it has tracked edits)
#   --branch N   also set refs/heads/N on the box (default: the commit-ish itself if it is a local branch name)
#   --keep N     after checkout keep only the newest N snapshots (default 10 / $WZ_KEEP_SNAPSHOTS; 0 = no gc; in-use kept)
# Prints WZ_TREE=<remote path> and verifies the remote tree SHA equals the local one.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=wz1080_common.sh
source "$HERE/wz1080_common.sh"

REV=""; MODE=wt; BRANCH=""; KEEP="${WZ_KEEP_SNAPSHOTS:-10}"; CHECKOUT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --main) MODE=main; shift;;
    --branch) BRANCH="$2"; shift 2;;
    --keep) KEEP="$2"; shift 2;;
    --no-checkout) CHECKOUT=0; shift;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    -*) echo "unknown option $1" >&2; exit 2;;
    *) REV="$1"; shift;;
  esac
done
[[ -n "$REV" ]] || { echo "usage: $0 <commit-ish> [--main] [--branch NAME] [--keep N]" >&2; exit 2; }

G=(git -C "$WZ_LOCAL_REPO")
SHA="$("${G[@]}" rev-parse --verify "${REV}^{commit}")"
LOCAL_TREE="$("${G[@]}" rev-parse "${SHA}^{tree}")"
if [[ -z "$BRANCH" ]] && "${G[@]}" show-ref --verify -q "refs/heads/$REV"; then BRANCH="$REV"; fi
echo "[wz1080_sync] $REV = ${SHA:0:12} (tree ${LOCAL_TREE:0:12})${BRANCH:+ branch $BRANCH}"

# The remote git helper travels inline with every call (no dependency on the box's tools copy, which other lanes'
# script versions must not flip) and all git operations on the box are serialized (several lanes sync concurrently).
rg() { wz_ssh "mkdir -p $WZ_R_SYNC && flock $WZ_R_SYNC/.git.lock bash -s -- $*" < "$HERE/wz1080_remote_git.sh"; }

mapfile -t STATE < <(rg state "$SHA")
HEAD_LINE="${STATE[0]}"; echo "[wz1080_sync] box: $HEAD_LINE"
if [[ "$HEAD_LINE" == *"have=1"* ]]; then
  [[ -n "$BRANCH" ]] && rg branch "$SHA" "$BRANCH"
else
  wz_wait_quiet
  BASIS=()
  for s in "${STATE[@]:1}"; do
    [[ "$s" =~ ^[0-9a-f]{40}$ ]] && "${G[@]}" cat-file -e "${s}^{commit}" 2>/dev/null && BASIS+=("$s")
  done
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)-${SHA:0:12}"
  # Throwaway dir = mktemp child of a fixed, canonical root; cleanup only through wz_rm_confined (validated path).
  mkdir -p "$WZ_LOCAL_LANE/bundles"
  BUNDLE_ROOT="$(realpath -e -- "$WZ_LOCAL_LANE/bundles")"
  TMPD_RE='^sync\.[A-Za-z0-9]{8}$'
  TMPD="$(mktemp -d "$BUNDLE_ROOT/sync.XXXXXXXX")"
  trap 'wz_rm_confined "$BUNDLE_ROOT" "$TMPD" "$TMPD_RE"' EXIT
  # Private ref namespace in a throwaway repo that borrows the main object store (alternates):
  # the main repo's refs are never touched (git bundle needs a ref name for the tip).
  git init -q --bare "$TMPD/r.git"
  echo "$("${G[@]}" rev-parse --path-format=absolute --git-common-dir)/objects" > "$TMPD/r.git/objects/info/alternates"
  git --git-dir="$TMPD/r.git" update-ref "refs/wz/sync/${SHA:0:12}" "$SHA"
  echo "[wz1080_sync] bundling (basis: ${#BASIS[@]} known commits on the box)"
  if (( ${#BASIS[@]} )); then
    wz_local_heavy git --git-dir="$TMPD/r.git" bundle create -q "$TMPD/x.bundle" "refs/wz/sync/${SHA:0:12}" --not "${BASIS[@]}"
  else
    wz_local_heavy git --git-dir="$TMPD/r.git" bundle create -q "$TMPD/x.bundle" "refs/wz/sync/${SHA:0:12}"
  fi
  echo "[wz1080_sync] bundle $(du -h "$TMPD/x.bundle" | cut -f1); transferring"
  wz_rsync --partial "$TMPD/x.bundle" "$WZ_HOST:$WZ_R_SYNC/$STAMP.bundle"
  rg import "$WZ_R_SYNC/$STAMP.bundle" "$SHA" "$BRANCH"
  wz_rm_confined "$BUNDLE_ROOT" "$TMPD" "$TMPD_RE"; trap - EXIT
fi

(( CHECKOUT )) || exit 0
OUT="$(rg checkout "$SHA" "$MODE")"
echo "$OUT"
REMOTE_TREE="$(sed -n 's/.*WZ_TREE_SHA=\([0-9a-f]*\).*/\1/p' <<<"$OUT")"
if [[ "$REMOTE_TREE" != "$LOCAL_TREE" ]]; then
  echo "[wz1080_sync] TREE MISMATCH local $LOCAL_TREE vs box $REMOTE_TREE" >&2; exit 5
fi
echo "[wz1080_sync] OK tree identical (${LOCAL_TREE:0:12})"
if [[ "$MODE" == wt && "$KEEP" -gt 0 ]]; then rg gc "$KEEP"; fi
