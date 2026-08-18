#!/bin/zsh
set -euo pipefail

umask 077

refresh_args=()
if [[ "${1:-}" == "--force" && "$#" -eq 1 ]]; then
  refresh_args+=(--force)
elif [[ "$#" -ne 0 ]]; then
  print -u2 "Usage: run-public-refresh.sh [--force]"
  exit 64
fi

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
public_root="${worker_root}/public-packs"
runtime_root="${worker_root}/runtime"
python="${worker_root}/venv/bin/python"
"${repo_root}/ops/verify-public-refresh-env.sh" "${repo_root}" "${worker_root}/venv"
oe_inbox="${worker_root}/oe-inbox"
cycle_id="$("${python}" -c 'from datetime import datetime; now=datetime.now(); print(f"{now:%Y%m%d}T{now.hour // 6 * 6:02d}0000")')"
run_root="${runtime_root}/data/lol/runtime/cycles/${cycle_id}"
worker_lock="${runtime_root}/data/lol/runtime/public-refresh-worker.lock"
oe_year="2026"
oe_name="${oe_year}_LoL_esports_match_data_from_OraclesElixir.csv"
oe_candidate="${oe_inbox}/${oe_name}"
oe_download_url="https://drive.google.com/uc?export=download&id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"

export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SCRYGLASS_PUBLIC_RELEASE=1
export SCRYGLASS_PUBLISH_ORIGIN="https://scryglass.xyz"
export SCRYGLASS_PUBLICATION_BACKEND="supabase"
export SCRYGLASS_RUNTIME_ROOT="${runtime_root}"
export SCRYGLASS_SUPABASE_URL="https://uytblwbtkwuukbbrugdi.supabase.co"
export SCRYGLASS_REFRESH_ATTEMPTS=3
export SCRYGLASS_STEP_TIMEOUT_MINUTES=15
export SCRYGLASS_STALE_AFTER_HOURS=12
export SCRYGLASS_OE_BROWSER_REFRESHED=1
export SCRYGLASS_OE_DATABASE_REFRESHED=1
# Ask Python where the annual CSV lives instead of hardcoding it. The Python
# side resolves RAW_OE_DIR to the Worker inbox when it exists and falls back to
# "${runtime_root}/data/lol/warehouse/raw" otherwise, which is exactly what this
# script used to hardcode. Two independent resolvers had already drifted: the
# downloader accepted the fresh inbox CSV and wrote a receipt binding its bytes,
# then this script handed the importer the stale warehouse copy and the receipt
# check refused the run.
raw_oe_dir="$(cd "${repo_root}" && "${python}" -m lol_kills.etl.paths --raw-oe-dir)"
if [[ -z "${raw_oe_dir}" || ! -d "${raw_oe_dir}" ]]; then
  print -u2 "Could not resolve the Oracle's Elixir source directory."
  exit 78
fi
oe_csv="${raw_oe_dir}/${oe_name}"
if [[ ! -f "${oe_csv}" ]]; then
  print -u2 "Oracle's Elixir annual CSV is missing at ${oe_csv}."
  exit 78
fi

real_worker_commit="$(/usr/bin/git -C "${repo_root}" rev-parse --verify HEAD)"
if [[ ! "${SCRYGLASS_WORKER_COMMIT:-}" =~ '^[0-9a-f]{40}$' ]]; then
  print -u2 "SCRYGLASS_WORKER_COMMIT must name the tested worker commit."
  exit 78
fi
if [[ "${SCRYGLASS_WORKER_COMMIT}" != "${real_worker_commit}" ]]; then
  print -u2 "The worker HEAD differs from SCRYGLASS_WORKER_COMMIT."
  exit 78
fi
if [[ -n "$(/usr/bin/git -C "${repo_root}" status --porcelain=v1 --untracked-files=normal)" ]]; then
  print -u2 "The worker checkout contains uncommitted files."
  exit 78
fi
export SCRYGLASS_WORKER_COMMIT
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
export SCRYGLASS_DIAGNOSTIC_TOKEN="$(
  /usr/bin/security find-generic-password \
    -a scryglass-public-worker \
    -s scryglass-diagnostic-token \
    -w
)"

mkdir -p "${public_root}" "${worker_root}/logs" "${oe_inbox}" "${run_root}"
if ! /usr/bin/shlock -p "$$" -f "${worker_lock}"; then
  print -u2 "Another Scryglass public refresh owns ${worker_lock}."
  exit 75
fi
trap '/bin/rm -f "${worker_lock}"' EXIT HUP INT TERM

source_receipt="${run_root}/accepted-source.json"
import_receipt="${run_root}/accepted-import.json"
patch_receipt_catalog="${run_root}/riot-patch-receipts.json"
patch_receipt_args=()
if [[ -f "${patch_receipt_catalog}" ]]; then
  patch_receipt_args=(--patch-receipts "${patch_receipt_catalog}")
fi
export SCRYGLASS_ACCEPTED_SOURCE_RECEIPT="${source_receipt}"
export SCRYGLASS_ACCEPTED_IMPORT_RECEIPT="${import_receipt}"

/usr/bin/git -C "${repo_root}" archive HEAD \
  data/lol/v2 \
  | /usr/bin/tar -x -C "${runtime_root}"

cd "${repo_root}"
# launchd can start this script with a working directory outside the checkout.
# Keep module discovery bound to the attested worker checkout.
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

resume_cycle=0
if [[ ! -f "${patch_receipt_catalog}" && -f "${source_receipt}" && -f "${import_receipt}" ]]; then
  if "${python}" -m lol_kills.etl.oe_database \
    --csv "${oe_csv}" \
    --year "${oe_year}" \
    --parquet-dir "${runtime_root}/data/lol/warehouse/parquet" \
    --source-receipt "${source_receipt}" \
    "${patch_receipt_args[@]}" \
    --result-output "${import_receipt}" \
    --validate-only >/dev/null; then
    resume_cycle=1
  fi
fi

if [[ "${resume_cycle}" -eq 0 ]]; then
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

  "${python}" -m lol_kills.etl.oe_ingest \
    --install-browser-candidate "${oe_candidate}" \
    --year "${oe_year}" \
    --receipt-output "${source_receipt}"
  "${python}" -m lol_kills.etl.oe_database \
    --csv "${oe_csv}" \
    --year "${oe_year}" \
    --parquet-dir "${runtime_root}/data/lol/warehouse/parquet" \
    --source-receipt "${source_receipt}" \
    "${patch_receipt_args[@]}" \
    --result-output "${import_receipt}"
fi

exec "${python}" -m lol_kills.public_refresh \
  --root "${repo_root}" \
  --public-root "${public_root}" \
  --once \
  "${refresh_args[@]}"
