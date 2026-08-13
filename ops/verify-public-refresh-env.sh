#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: verify-public-refresh-env.sh REPO_ROOT VENV_PATH" >&2
  exit 64
fi

repo_root=$1
venv_path=$2
lock_path="${repo_root}/requirements.lock"
marker_path="${venv_path}/.scryglass-requirements-lock.sha256"

if [ ! -f "${lock_path}" ] || [ ! -f "${marker_path}" ]; then
  echo "The worker environment has no requirements-lock attestation." >&2
  exit 78
fi

expected=$(/usr/bin/shasum -a 256 "${lock_path}" | /usr/bin/awk '{print $1}')
IFS= read -r installed < "${marker_path}"
if [ "${installed}" != "${expected}" ]; then
  echo "The worker environment does not match requirements.lock." >&2
  exit 78
fi

"${venv_path}/bin/python" -m pip check >/dev/null
