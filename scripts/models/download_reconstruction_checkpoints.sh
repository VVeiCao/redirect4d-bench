#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R4D_ROOT="${REPO_ROOT}/reconstruction/redirect4d"

exec bash "${R4D_ROOT}/download_checkpoints.sh" "${1:-required}"
