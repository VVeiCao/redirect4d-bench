#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-redirect4d-bench}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

git -C "${REPO_ROOT}" submodule update --init --recursive
conda env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml" --prune
