#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-redirect4d-sam3}"
PYTHON_VERSION="3.12"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAM3_ROOT="${REPO_ROOT}/third_party/sam3"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

git -C "${REPO_ROOT}" submodule update --init --recursive third_party/sam3

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

pip install -U pip "setuptools<81" wheel
pip install "torch==${TORCH_VERSION}" torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -e "${SAM3_ROOT}" opencv-python imageio-ffmpeg einops pycocotools psutil "huggingface_hub[cli]"
pip install -e "${REPO_ROOT}" --no-deps

python - <<'PY'
import cv2
import imageio_ffmpeg
import torch
from sam3.model_builder import build_sam3_video_model

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("opencv:", cv2.__version__)
print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())
print("sam3: OK")
PY

cat <<EOF

Activate with:
  conda activate ${ENV_NAME}

Run mask refinement with:
  python scripts/pseudo_gt/refine_target_masks_sam3.py --help

SAM3 checkpoints are downloaded through Hugging Face on first use. If your
checkpoint access requires authentication, run:
  hf auth login
EOF
