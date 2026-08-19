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

# Curate PATH before anything else runs: the self-healing preamble below shells
# out to git, and launchd hands this script a minimal environment.
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

worker_root="${SCRYGLASS_WORKER_ROOT:-${HOME}/Library/Application Support/Scryglass Worker}"
repo_root="${worker_root}/repo"
public_root="${worker_root}/public-packs"
runtime_root="${worker_root}/runtime"
python="${worker_root}/venv/bin/python"
worker_lock="${runtime_root}/data/lol/runtime/public-refresh-worker.lock"
launcher_installed="${worker_root}/run-public-refresh.sh"
launcher_source="${repo_root}/ops/launchd/run-public-refresh.sh"

# Reap a lock whose owner is gone, before anything else touches the worker.
#
# /usr/bin/shlock is documented to reclaim a lock held by a dead process, but
# the macOS implementation gates that takeover on the lock file's ctime and
# refuses whenever that ctime is not strictly older than the invocation. A lock
# left behind by a killed run therefore keeps blocking every later run as soon
# as anything restamps its ctime: a same-second restart, a chmod, a copy, a
# restore from backup. Runs were wedged this way and had to be unblocked by
# hand. Decide staleness here so the cycle never depends on that heuristic.
# A lock whose owner is still running is left alone and still refuses the run.
if [[ -f "${worker_lock}" ]]; then
  # A lock that cannot be read cannot be proven stale, so it is refused rather
  # than reaped: removing it might remove a live owner's lock.
  if ! lock_owner="$(/bin/cat "${worker_lock}" 2>/dev/null)"; then
    print -u2 "Cannot read ${worker_lock} to identify its owner."
    exit 75
  fi
  lock_owner="${lock_owner//[[:space:]]/}"
  if [[ "${lock_owner}" =~ '^[1-9][0-9]*$' ]]; then
    # Never signal pid 0: kill(0, 0) addresses the whole process group and
    # would report a dead owner as live. The pattern above excludes it.
    if /bin/kill -0 "${lock_owner}" 2>/dev/null; then
      print -u2 "Another Scryglass public refresh owns ${worker_lock} (pid ${lock_owner})."
      exit 75
    fi
    /bin/rm -f "${worker_lock}"
    print -r -- "public-refresh: reaped stale lock from pid ${lock_owner}"
  else
    /bin/rm -f "${worker_lock}"
    print -r -- "public-refresh: reaped unreadable lock ${worker_lock}"
  fi
fi

# Run a command under a wall-clock bound. macOS ships no timeout(1), and a
# stalled fetch must never hold the schedule open. Returns the command's own
# status, or 124 when the bound is reached.
bounded_run () {
  local timeout_seconds=$1
  shift
  local waited=0
  # Not named "status": zsh reserves that as a read-only alias for $?.
  local job_status=0
  "$@" &
  local job_pid=$!
  while (( waited < timeout_seconds )); do
    /bin/kill -0 "${job_pid}" 2>/dev/null || break
    /bin/sleep 1
    (( waited += 1 ))
  done
  if /bin/kill -0 "${job_pid}" 2>/dev/null; then
    /bin/kill -TERM "${job_pid}" 2>/dev/null || true
    /bin/sleep 1
    /bin/kill -KILL "${job_pid}" 2>/dev/null || true
    wait "${job_pid}" 2>/dev/null || true
    return 124
  fi
  if wait "${job_pid}"; then
    job_status=0
  else
    job_status=$?
  fi
  return ${job_status}
}

# Sync the worker checkout to origin/main.
#
# Pinning the tested commit by hand took four steps -- sync the repo, copy this
# launcher, render the plist, reload the agent -- and skipping any one of them
# failed the run against refresh_ledger.worker_commit after an hour of work.
# main is the tested branch: merging into it requires green required checks and
# resolved review threads. So the launcher moves itself to origin/main and
# derives the commit from the checkout instead of trusting an injected pin.
#
# Fetch is fail-open and consistency is fail-closed: a network blip must not
# cancel a refresh, but a checkout that disagrees with itself must still stop.
fetch_ok=0
if bounded_run 120 /usr/bin/env GIT_TERMINAL_PROMPT=0 /usr/bin/git -C "${repo_root}" \
  -c http.lowSpeedLimit=1024 \
  -c http.lowSpeedTime=30 \
  fetch --quiet origin '+refs/heads/main:refs/remotes/origin/main'; then
  fetch_ok=1
else
  print -u2 "public-refresh: could not fetch origin/main; continuing on the current checkout."
fi

sync_target=""
if (( fetch_ok )); then
  sync_target="$(/usr/bin/git -C "${repo_root}" rev-parse --verify --quiet refs/remotes/origin/main || true)"
fi
head_commit="$(/usr/bin/git -C "${repo_root}" rev-parse --verify HEAD)"
if [[ -n "${sync_target}" && "${sync_target}" != "${head_commit}" ]]; then
  # Never reset over local work. A dirty checkout keeps its commit and still
  # fails the mandatory check below, which is the loud outcome an operator who
  # is editing the worker in place needs to see.
  if [[ -n "$(/usr/bin/git -C "${repo_root}" status --porcelain=v1 --untracked-files=normal)" ]]; then
    print -u2 "public-refresh: worker checkout is dirty; not syncing ${head_commit} to ${sync_target}."
  else
    print -r -- "public-refresh: syncing worker checkout ${head_commit} to ${sync_target}"
    /usr/bin/git -C "${repo_root}" reset --hard --quiet "${sync_target}"
  fi
fi

# Adopt the launcher that the synced commit ships. launchd invokes the copy at
# the worker root, so an improvement to this file would otherwise never reach a
# scheduled run. Re-exec at most once: if the copy still differs afterwards the
# install is broken, and looping would be worse than stopping.
if [[ -f "${launcher_source}" ]] && ! /usr/bin/cmp -s "${launcher_source}" "${launcher_installed}"; then
  if [[ "${SCRYGLASS_LAUNCHER_REEXEC:-}" == "1" ]]; then
    print -u2 "The installed launcher still differs from ${launcher_source} after one re-exec."
    exit 78
  fi
  /bin/cp -f "${launcher_source}" "${launcher_installed}"
  /bin/chmod 700 "${launcher_installed}"
  print -r -- "public-refresh: installed launcher from ${launcher_source}; re-executing"
  export SCRYGLASS_LAUNCHER_REEXEC=1
  exec "${launcher_installed}" "$@"
fi

# Bind the run to a commit. SCRYGLASS_WORKER_COMMIT is still honoured when it is
# set, so an operator can pin a manual test run to an exact commit. When it is
# unset the launcher derives it from the synced checkout, which makes the
# refresh_ledger.worker_commit check hold by construction rather than by an
# operator remembering to re-render the launch agent. The clean-checkout
# requirement is mandatory in both modes.
real_worker_commit="$(/usr/bin/git -C "${repo_root}" rev-parse --verify HEAD)"
if [[ -n "${SCRYGLASS_WORKER_COMMIT:-}" ]]; then
  if [[ ! "${SCRYGLASS_WORKER_COMMIT}" =~ '^[0-9a-f]{40}$' ]]; then
    print -u2 "SCRYGLASS_WORKER_COMMIT must name the tested worker commit."
    exit 78
  fi
  if [[ "${SCRYGLASS_WORKER_COMMIT}" != "${real_worker_commit}" ]]; then
    print -u2 "The worker HEAD differs from SCRYGLASS_WORKER_COMMIT."
    exit 78
  fi
else
  SCRYGLASS_WORKER_COMMIT="${real_worker_commit}"
fi
if [[ -n "$(/usr/bin/git -C "${repo_root}" status --porcelain=v1 --untracked-files=normal)" ]]; then
  print -u2 "The worker checkout contains uncommitted files."
  exit 78
fi
export SCRYGLASS_WORKER_COMMIT

"${repo_root}/ops/verify-public-refresh-env.sh" "${repo_root}" "${worker_root}/venv"
oe_inbox="${worker_root}/oe-inbox"
cycle_id="$("${python}" -c 'from datetime import datetime; now=datetime.now(); print(f"{now:%Y%m%d}T{now.hour // 6 * 6:02d}0000")')"
run_root="${runtime_root}/data/lol/runtime/cycles/${cycle_id}"
oe_year="2026"
oe_name="${oe_year}_LoL_esports_match_data_from_OraclesElixir.csv"
oe_candidate="${oe_inbox}/${oe_name}"
# Mirrors Path.with_suffix(".part") in lol_kills.etl.oe_fetch: the staging file
# the headless downloader writes before its atomic rename. A run killed by the
# wall-clock bound below can leave one behind; it never matches the annual-CSV
# glob, and the next fetch removes it, but the Brave fallback clears it too.
oe_partial="${oe_candidate:r}.part"
oe_download_url="https://drive.google.com/uc?export=download&id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"

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
if [[ -z "${raw_oe_dir}" ]]; then
  print -u2 "Could not resolve the Oracle's Elixir source directory."
  exit 78
fi
# Create the directory rather than requiring it: on a fresh worker the download
# below is what first populates it.
/bin/mkdir -p "${raw_oe_dir}" || exit 78
oe_csv="${raw_oe_dir}/${oe_name}"
# Deliberately NOT asserting that ${oe_csv} exists here. The browser-download
# block further down is what creates it, and it deletes the candidate before
# opening Brave. Failing here would wedge a fresh worker permanently, and would
# also make a timed-out download unrecoverable without hand-placing a CSV. A
# genuinely missing file is still caught downstream by _validate_oe_csv and by
# the source-receipt check, which fail loudly and name the path.

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
  # Headless first; Brave is the last resort.
  #
  # This block used to open Brave on every single run and then poll the
  # browser's download directory for a stable file size. That tied an
  # unattended, scheduled job to a logged-in GUI session: no desktop, no CSV,
  # no refresh. The annual export is a public Google Drive object, so an
  # anonymous HTTPS GET fetches the same bytes with no GUI at all.
  #
  # lol_kills.etl.oe_fetch is fail-closed about what it will write: Drive
  # answers a quota-blocked or consent-gated request with HTTP 200 and a small
  # HTML page, and the fetcher refuses to let that reach ${oe_candidate}. It
  # exits 75 for "blocked, not broken" and non-zero otherwise, which is the
  # distinction this ladder needs. Every downstream check is unchanged: the
  # bytes still go through _validate_oe_csv and the source-receipt binding.
  # Fetch into a hidden staging file, NEVER the accepted CSV's own path. On
  # the worker RAW_OE_DIR resolves to this same inbox, so ${oe_candidate} IS
  # the previously accepted annual CSV: writing the fetch result there would
  # destroy the known-good file before install_browser_download has validated
  # the new bytes, and a regression (rotated file serving an older export, or
  # a wide malformed CSV) would then be compared against itself and installed.
  # The dot prefix keeps the staging file out of the annual-CSV glob.
  oe_headless_stage="${oe_inbox}/.headless-${oe_name}"
  oe_fetch_evidence="${run_root}/oe-headless-fetch.json"
  /bin/mkdir -p "${run_root}"
  oe_fetch_status=0
  bounded_run 300 "${python}" -m lol_kills.etl.oe_fetch \
    --year "${oe_year}" \
    --destination "${oe_headless_stage}" > "${oe_fetch_evidence}" || oe_fetch_status=$?

  oe_install_source="${oe_headless_stage}"
  need_browser_download=0
  if (( oe_fetch_status == 0 )); then
    print -r -- "public-refresh: headless Oracle's Elixir download wrote ${oe_headless_stage}"
    export SCRYGLASS_OE_TRANSPORT="anonymous_https_drive_download"
    # The pinned id can rotate; when the folder re-resolution supplied the
    # bytes, the receipt must name the file that ACTUALLY served them, not the
    # pinned object that failed. The fetcher printed its evidence as JSON.
    oe_resolved_id="$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("drive_file_id") or "")' "${oe_fetch_evidence}" 2>/dev/null || true)"
    oe_resolved_url="$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("url") or "")' "${oe_fetch_evidence}" 2>/dev/null || true)"
    if [[ -n "${oe_resolved_id}" ]]; then
      export SCRYGLASS_OE_DRIVE_FILE_ID="${oe_resolved_id}"
    fi
    if [[ -n "${oe_resolved_url}" ]]; then
      export SCRYGLASS_OE_LOCATOR="${oe_resolved_url}"
    fi
  elif (( oe_fetch_status == 75 )); then
    # A Drive quota block leaves the previous download in place, and that file
    # is real data - only its age is in question. Reuse it while it is inside
    # the same limit the ingest applies to a stale source, so a quota block
    # costs a few hours of freshness instead of reintroducing the GUI
    # dependency. The importer dedupes unchanged bytes, so re-presenting the
    # same file is cheap and records "unchanged" rather than a false refresh.
    oe_max_age_days="$(
      "${python}" -c \
        'from lol_kills.etl.oe_ingest import OE_SOURCE_MAX_AGE_DAYS; print(int(OE_SOURCE_MAX_AGE_DAYS))' \
        2>/dev/null || true
    )"
    if [[ ! "${oe_max_age_days}" =~ '^[1-9][0-9]*$' ]]; then
      # Import failed; do not invent a longer grace period than the code's.
      oe_max_age_days=3
    fi
    oe_fresh_after=$(( $(/bin/date +%s) - oe_max_age_days * 86400 ))
    oe_existing_mtime=0
    if [[ -f "${oe_csv}" ]]; then
      oe_existing_mtime="$(/usr/bin/stat -f %m "${oe_csv}" 2>/dev/null || print -r -- 0)"
    fi
    if (( oe_existing_mtime > oe_fresh_after )); then
      print -r -- "public-refresh: headless blocked by quota; existing CSV is fresh enough (${oe_csv}, within ${oe_max_age_days} days) - proceeding without Brave"
      oe_install_source="${oe_csv}"
      export SCRYGLASS_OE_TRANSPORT="cached_inbox_reuse"
    else
      print -u2 "public-refresh: headless blocked by quota and ${oe_csv} is missing or older than ${oe_max_age_days} days; falling back to the Brave download."
      need_browser_download=1
    fi
  else
    print -u2 "public-refresh: headless Oracle's Elixir download failed (status ${oe_fetch_status}); falling back to the Brave download."
    need_browser_download=1
  fi

  if (( need_browser_download )); then
    export SCRYGLASS_OE_TRANSPORT="brave_origin_browser_download"
    oe_install_source="${oe_candidate}"
    rm -f "${oe_candidate}" "${oe_candidate}.crdownload" "${oe_partial}" "${oe_headless_stage}"
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
  fi

  "${python}" -m lol_kills.etl.oe_ingest \
    --install-browser-candidate "${oe_install_source}" \
    --year "${oe_year}" \
    --receipt-output "${source_receipt}"
  /bin/rm -f "${oe_headless_stage}"
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
