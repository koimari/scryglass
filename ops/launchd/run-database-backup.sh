#!/bin/zsh
set -euo pipefail

umask 077

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
python="${worker_root}/venv/bin/python"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SCRYGLASS_DATABASE_URL="$(
  /usr/bin/security find-generic-password \
    -a scryglass-public-worker \
    -s scryglass-database-url \
    -w
)"

cd "${repo_root}"
exec "${python}" -m lol_kills.database_backup \
  --destination "${worker_root}/backups/postgres"
