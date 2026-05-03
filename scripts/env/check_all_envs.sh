#!/usr/bin/env bash
set -euo pipefail

BENCH_ENV="${BENCH_ENV:-redirect4d-bench}"
RECON_ENV="${RECON_ENV:-redirect4d-recon}"
SAM3_ENV="${SAM3_ENV:-redirect4d-sam3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

echo "[check] ${BENCH_ENV}"
conda run -n "${BENCH_ENV}" python "${REPO_ROOT}/scripts/dev/check_environment.py" --profile bench

echo
echo "[check] ${RECON_ENV}"
conda run -n "${RECON_ENV}" python "${REPO_ROOT}/scripts/dev/check_environment.py" --profile reconstruction

echo
echo "[check] ${SAM3_ENV}"
conda run -n "${SAM3_ENV}" python "${REPO_ROOT}/scripts/dev/check_environment.py" --profile sam3

echo
echo "[ok] all Redirect4D-Bench environments passed import checks"
