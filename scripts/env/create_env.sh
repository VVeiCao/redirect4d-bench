#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-redirect4d-bench}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_CHECK="${RUN_CHECK:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

git -C "${REPO_ROOT}" submodule update --init --recursive

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists."
  echo "Refreshing editable package and Python dependencies..."
  conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"
else
  conda env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml"
fi

if [[ "${RUN_CHECK}" != "0" ]]; then
  echo
  echo "Checking base environment..."
  conda run -n "${ENV_NAME}" python "${REPO_ROOT}/scripts/dev/check_environment.py" --profile bench
fi

echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "To skip the automatic check, run:"
echo "  RUN_CHECK=0 bash scripts/env/create_env.sh ${ENV_NAME}"
