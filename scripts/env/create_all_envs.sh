#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

git -C "${REPO_ROOT}" submodule update --init --recursive

"${REPO_ROOT}/scripts/env/create_env.sh" redirect4d-bench
"${REPO_ROOT}/scripts/env/create_reconstruction_env.sh" redirect4d-recon
"${REPO_ROOT}/scripts/env/create_sam3_env.sh" redirect4d-sam3

cat <<'EOF'

All Redirect4D-Bench environments were prepared:
  - redirect4d-bench: metadata, staging, evaluation wrappers
  - redirect4d-recon: VIPE + LyRA + NoOpt reconstruction and rendering
  - redirect4d-sam3: SAM3 target-mask refinement

Next:
  bash scripts/models/download_reconstruction_checkpoints.sh required
  bash scripts/env/check_all_envs.sh
EOF
