#!/bin/zsh
set -euo pipefail

umask 077

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
python="${worker_root}/venv/bin/python"

export PATH="/opt/homebrew/opt/libpq/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SCRYGLASS_PG_DUMP="/opt/homebrew/opt/libpq/bin/pg_dump"
export SCRYGLASS_PG_RESTORE="/opt/homebrew/opt/libpq/bin/pg_restore"

cd "${repo_root}"
exec "${python}" -m lol_kills.database_backup \
  --destination "${worker_root}/backups/postgres" \
  --supabase-workdir "${repo_root}"
