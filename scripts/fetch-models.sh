#!/usr/bin/env bash
# fetch-models.sh — serial, resumable HuggingFace downloader for this machine's network.
#
# Why not `hf download`: measured 2026-08-09, ~75% of HTTPS connections to
# huggingface.co are reset mid-handshake here (WinError 10054 / curl 35), and
# huggingface_hub does not retry connection resets hard enough. The failure is
# client- and User-Agent-independent, and genuine parallelism makes it worse.
# So: one file at a time, resume (-C -), retry everything, and treat a stalled
# transfer (<2 KB/s for 20 s) as a failure worth restarting. See CLAUDE.md.
#
# Usage: bash scripts/fetch-models.sh [manifest]
# Manifest lines:  <repo_id> <dest_dir> [file_glob]
#   file_glob absent -> whole repo. Comments with '#', blank lines ignored.

set -uo pipefail

MANIFEST="${1:-scripts/models-wanted.txt}"
LOG="${FETCH_LOG:-$HOME/models/.fetch.log}"
mkdir -p "$(dirname "$LOG")"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

CURL=(curl -sS -L --fail
      -C -
      --retry 30 --retry-all-errors --retry-delay 1
      --speed-limit 2000 --speed-time 20
      --connect-timeout 20)

# A gated repo answers 401 with X-Error-Code: GatedRepo, and the API will
# still LIST it happily -- so the failure looks like a download problem
# rather than an access one. `hf auth login` leaves a token in the standard
# cache path; read it when HF_TOKEN is not exported, which is what the
# huggingface CLI itself does. The value is never printed.
if [ -z "${HF_TOKEN:-}" ]; then
  for _t in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
    [ -f "$_t" ] && HF_TOKEN=$(tr -d "
" < "$_t") && break
  done
fi
[ -n "${HF_TOKEN:-}" ] && CURL+=(-H "Authorization: Bearer $HF_TOKEN")

# Print "path<TAB>size" for every file in a repo, optionally filtered by glob.
list_repo() {
  local repo="$1" glob="${2:-}"
  # -L matters. A renamed repo answers the API with a 307, and without it
  # the listing came back EMPTY and the model was reported as "skipping"
  # rather than as moved: Comfy-Org/flux2-klein redirects to a repo with a
  # different name and three files went silently undownloaded. The download
  # URLs below always followed redirects; only the listing was blind.
  curl -sSL --retry 10 --retry-all-errors --retry-delay 1 -m 60 \
       ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
       "https://huggingface.co/api/models/$repo/tree/main?recursive=1" |
  GLOB="$glob" python -c '
import sys, json, os, fnmatch
glob = os.environ.get("GLOB", "")
for f in json.load(sys.stdin):
    if f.get("type") != "file":
        continue
    p = f["path"]
    if glob and not fnmatch.fnmatch(os.path.basename(p), glob):
        continue
    print(p, f.get("size", 0), sep="\t")
'
}

total_start=$(date +%s)
say "=== fetch-models start (manifest: $MANIFEST) ==="

while read -r repo dest glob; do
  case "$repo" in ''|'#'*) continue;; esac
  dest="${dest/#\~/$HOME}"
  say ""
  say ">>> $repo -> $dest ${glob:+(files: $glob)}"
  mkdir -p "$dest"

  listing="$(list_repo "$repo" "${glob:-}")"
  if [ -z "$listing" ]; then
    say "!!! could not list $repo (API unreachable or empty match) — skipping"
    continue
  fi

  n=0
  while IFS=$'\t' read -r path size; do
    [ -z "$path" ] && continue
    n=$((n+1))
    # A basename glob means the caller wants THOSE FILES AT dest, not the
    # repo tree rebuilt underneath it. Comfy-Org nests everything under
    # split_files/, so preserving the path put the model in
    # models/diffusion_models/split_files/diffusion_models/ -- invisible to
    # ComfyUI, and it restarted a 6 GB download because the resume target
    # was a different path than the one already on disk.
    if [ -n "${glob:-}" ]; then out="$dest/$(basename "$path")"; else out="$dest/$path"; fi
    mkdir -p "$(dirname "$out")"

    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$have" -eq "$size" ] && [ "$size" -gt 0 ]; then
      say "  ok   $path ($(numfmt --to=iec "$size" 2>/dev/null || echo "$size"))"
      continue
    fi

    say "  get  $path ($(numfmt --to=iec "$size" 2>/dev/null || echo "$size")${have:+, have $(numfmt --to=iec "$have" 2>/dev/null || echo "$have")})"
    t0=$(date +%s)
    if "${CURL[@]}" -o "$out" \
        "https://huggingface.co/$repo/resolve/main/$path?download=true" 2>>"$LOG"; then
      got=$(stat -c %s "$out" 2>/dev/null || echo 0)
      dt=$(( $(date +%s) - t0 )); [ "$dt" -eq 0 ] && dt=1
      if [ "$size" -gt 0 ] && [ "$got" -ne "$size" ]; then
        say "  BAD  $path: $got != $size bytes — left in place, rerun to resume"
      else
        say "  done $path in ${dt}s ($(( got / dt / 1000 )) kB/s)"
      fi
    else
      say "  FAIL $path — left partial, rerun to resume"
    fi
  done <<< "$listing"
  say "<<< $repo: $n file(s)"
done < "$MANIFEST"

say ""
say "=== fetch-models done in $(( ($(date +%s) - total_start) / 60 )) min ==="
