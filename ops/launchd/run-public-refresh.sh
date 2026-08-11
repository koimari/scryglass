#!/bin/zsh
set -euo pipefail

umask 077

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
public_root="${worker_root}/public-packs"
python="${worker_root}/venv/bin/python"
oe_inbox="${worker_root}/oe-inbox"
oe_year="2026"
oe_name="${oe_year}_LoL_esports_match_data_from_OraclesElixir.csv"
oe_candidate="${oe_inbox}/${oe_name}"
oe_download_url="https://drive.google.com/uc?export=download&id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SCRYGLASS_PUBLIC_RELEASE=1
export SCRYGLASS_PUBLISH_ORIGIN="https://scryglass.xyz"
export SCRYGLASS_PUBLICATION_BACKEND="supabase"
export SCRYGLASS_SUPABASE_URL="https://uytblwbtkwuukbbrugdi.supabase.co"
export SCRYGLASS_REFRESH_ATTEMPTS=3
export SCRYGLASS_STEP_TIMEOUT_MINUTES=15
export SCRYGLASS_STALE_AFTER_HOURS=12
export SCRYGLASS_OE_BROWSER_REFRESHED=1
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

mkdir -p "${public_root}" "${worker_root}/logs" "${oe_inbox}"
rm -f "${oe_candidate}" "${oe_candidate}.crdownload"
/usr/bin/open -a "Brave Origin" "${oe_download_url}"

download_ready=0
for _attempt in {1..180}; do
  if [[ -f "${oe_candidate}" && ! -f "${oe_candidate}.crdownload" ]]; then
    first_size="$(/usr/bin/stat -f %z "${oe_candidate}")"
    /bin/sleep 2
    second_size="$(/usr/bin/stat -f %z "${oe_candidate}")"
    if [[ "${first_size}" -ge 10000 && "${first_size}" -eq "${second_size}" ]]; then
      download_ready=1
      break
    fi
  fi
  /bin/sleep 1
done

if [[ "${download_ready}" -ne 1 ]]; then
  print -u2 "Fresh Oracle's Elixir browser download did not arrive within 180 seconds."
  exit 1
fi

cd "${repo_root}"
"${python}" -m lol_kills.etl.oe_ingest \
  --install-browser-candidate "${oe_candidate}" \
  --year "${oe_year}"

exec "${python}" -m lol_kills.public_refresh \
  --root "${repo_root}" \
  --public-root "${public_root}" \
  --once
