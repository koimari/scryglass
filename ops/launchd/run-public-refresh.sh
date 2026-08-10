#!/bin/zsh
set -euo pipefail

umask 077

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
public_root="${worker_root}/public-packs"
python="${worker_root}/venv/bin/python"

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SCRYGLASS_PUBLIC_RELEASE=1
export SCRYGLASS_PUBLISH_ORIGIN="https://scryglass.xyz"
export SCRYGLASS_PUBLICATION_BACKEND="supabase"
export SCRYGLASS_SUPABASE_URL="https://uytblwbtkwuukbbrugdi.supabase.co"
export SCRYGLASS_REFRESH_ATTEMPTS=3
export SCRYGLASS_STEP_TIMEOUT_MINUTES=15
export SCRYGLASS_STALE_AFTER_HOURS=12
export SCRYGLASS_SUPABASE_SECRET_KEY="$(
  /usr/bin/security find-generic-password \
    -a scryglass-public-worker \
    -s scryglass-supabase-secret \
    -w
)"
export SCRYGLASS_DATA_PUBLISH_TOKEN="$(
  /usr/bin/security find-generic-password \
    -a scryglass-public-worker \
    -s scryglass-data-publish-token \
    -w
)"

mkdir -p "${public_root}" "${worker_root}/logs"
cd "${repo_root}"

exec "${python}" -m lol_kills.public_refresh \
  --root "${repo_root}" \
  --public-root "${public_root}" \
  --once
