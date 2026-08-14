#!/bin/sh
set -eu

umask 077

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: install-public-refresh-env.sh VENV_PATH [PYTHON]" >&2
  exit 64
fi

venv_path=$1
python_bin=${2:-python3}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
lock_path="${repo_root}/requirements.lock"

if [ ! -f "${lock_path}" ]; then
  echo "The hashed worker requirements lock is missing." >&2
  exit 78
fi

"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install \
  --disable-pip-version-check \
  --require-hashes \
  --only-binary=:all: \
  --requirement "${lock_path}"

lock_sha256=$(/usr/bin/shasum -a 256 "${lock_path}" | /usr/bin/awk '{print $1}')
marker=$(mktemp "${venv_path}/.scryglass-requirements-lock.sha256.XXXXXX")
trap 'rm -f "${marker}"' EXIT HUP INT TERM
printf '%s\n' "${lock_sha256}" > "${marker}"
chmod 600 "${marker}"
mv "${marker}" "${venv_path}/.scryglass-requirements-lock.sha256"
trap - EXIT HUP INT TERM
