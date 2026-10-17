#!/usr/bin/env bash
# wz1080_remote_git.sh — REMOTE side of wz1080_sync.sh (runs on the test box from ~/src/wrf_gpu2_env/tools/).
#   state <sha>                      -> "repo=0|1 have=0|1" then one known commit SHA per line (bundle basis)
#   import <bundle> <sha> [branch]   -> fetch bundle into refs/synced/<sha12>, optionally refs/heads/<branch>
#   checkout <sha> wt|main           -> detached snapshot (wt) or move the main clone; data overlay; prints WZ_TREE=
#   branch <sha> <name>              -> set refs/heads/<name>
#   gc <keep>                        -> keep the newest <keep> snapshots (never one that a process uses)
set -euo pipefail
cd "$HOME"
REPO="$HOME/src/wrf_gpu2"; WT="$HOME/src/wrf_gpu2_wt"; DATA="$HOME/src/wrf_gpu2_data"

overlay() {  # $1 = tree. Pristine WRF link (AGENTS rule) + git-ignored test fixtures (reflink copy, never overwrite).
  local t="$1"
  if [[ -d "$HOME/src/wrf_pristine" ]]; then
    mkdir -p "$t/data"; [[ -e "$t/data/wrf_pristine" ]] || ln -s "$HOME/src/wrf_pristine" "$t/data/wrf_pristine"
  fi
  if [[ -d "$DATA/overlay" ]]; then
    cp -a --reflink=auto --update=none "$DATA/overlay/." "$t/" 2>/dev/null || cp -a --reflink=auto -n "$DATA/overlay/." "$t/"
  fi
}

cmd="${1:-}"; shift || true
case "$cmd" in
  state)
    sha="$1"
    if [[ -d "$REPO/.git" ]]; then
      have=0; git -C "$REPO" cat-file -e "${sha}^{commit}" 2>/dev/null && have=1
      echo "repo=1 have=$have"
      git -C "$REPO" for-each-ref --format='%(objectname)' refs/synced refs/heads | sort -u
    else
      echo "repo=0 have=0"
    fi;;
  import)
    bundle="$(readlink -f "$1")"; sha="$2"; branch="${3:-}"
    trap 'rm -f "$bundle"' EXIT
    [[ -d "$REPO/.git" ]] || { mkdir -p "$(dirname "$REPO")"; git init -q "$REPO"; }
    git -C "$REPO" bundle verify -q "$bundle" >/dev/null 2>&1
    git -C "$REPO" fetch -q "$bundle" 'refs/wz/sync/*:refs/synced/*'
    git -C "$REPO" cat-file -e "${sha}^{commit}"
    [[ -n "$branch" ]] && git -C "$REPO" update-ref "refs/heads/$branch" "$sha"
    echo "imported ${sha:0:12}${branch:+ as $branch}";;
  branch)
    git -C "$REPO" update-ref "refs/heads/$2" "$1"; echo "branch $2 -> ${1:0:12}";;
  checkout)
    sha="$1"; mode="${2:-wt}"
    if [[ "$mode" == main ]]; then
      if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null | head -1)" ]]; then
        echo "REFUSED: $REPO has local modifications to tracked files; use a snapshot (wt) or clean it" >&2; exit 4
      fi
      git -C "$REPO" -c advice.detachedHead=false checkout -q --detach "$sha"
      tree="$REPO"
    else
      tree="$WT/${sha:0:12}"
      if [[ ! -d "$tree" ]]; then
        mkdir -p "$WT"
        git -C "$REPO" worktree add -q --detach "$tree" "$sha"
      fi
      touch "$tree"   # mtime = last use (gc order)
    fi
    overlay "$tree"
    echo "WZ_TREE=$tree"
    echo "WZ_HEAD=$(git -C "$tree" rev-parse HEAD) WZ_TREE_SHA=$(git -C "$tree" rev-parse 'HEAD^{tree}')";;
  gc)
    keep="${1:-6}"; n=0
    busy="$(ls -l /proc/[0-9]*/cwd 2>/dev/null | awk '{print $NF}' | sort -u)"
    for d in $(ls -1dt "$WT"/*/ 2>/dev/null); do
      d="${d%/}"; n=$((n+1)); (( n <= keep )) && continue
      if grep -q "^$d" <<<"$busy"; then echo "keep (in use): $d"; continue; fi
      git -C "$REPO" worktree remove --force "$d" && echo "removed $d"
    done
    git -C "$REPO" worktree prune;;
  *) echo "usage: $0 state|import|branch|checkout|gc ..." >&2; exit 2;;
esac
