#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="${1:-redirect4d-recon}"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.4.0"
CUDA_TAG="cu121"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R4D_ROOT="${REPO_ROOT}/reconstruction/redirect4d"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists."
else
  echo "Creating reconstruction environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

pip install -U pip "setuptools<81" wheel
pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==0.19.0" \
  "torchaudio==${PYTORCH_VERSION}"

pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  xformers==0.0.27.post1 --no-deps

conda install -c conda-forge ffmpeg -y
pip install -r "${R4D_ROOT}/requirements.txt"

git -C "${REPO_ROOT}" submodule update --init --recursive

pip install -e "${REPO_ROOT}" --no-deps
pip install -e "${REPO_ROOT}/third_party/DiffSynth-Studio" --no-deps
pip install -e "${REPO_ROOT}/third_party/generative-models" --no-deps

conda install -c "nvidia/label/cuda-12.1.1" cuda-nvcc cuda-cudart-dev cuda-cccl \
  libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev cuda-nvrtc-dev -y
conda install -c conda-forge gxx_linux-64=12 gcc_linux-64=12 -y
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
pip install -e "${REPO_ROOT}/third_party/vipe" --no-deps --no-build-isolation

pip install "git+https://github.com/microsoft/MoGe.git"
pip install gdown rerun-sdk python-pycg OpenEXR click \
  "git+https://github.com/heiwang1997/Depth-Anything-3.git@main"
pip install -e "git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata"
pip install "git+https://github.com/openai/CLIP.git"
pip install "transformers==4.57.6"
pip install accelerate
pip install --no-deps "git+https://github.com/facebookresearch/sam2.git"
pip install hydra-core iopath
pip install --no-deps "git+https://github.com/facebookresearch/vggt.git"
pip install webdataset "torchdata<0.8"
pip install --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
  --no-build-isolation

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), torch.version.cuda)
import xformers
print("xformers:", xformers.__version__)
import open3d
print("Open3D:", open3d.__version__)
import sgm
print("sgm: OK")
import diffsynth
print("diffsynth: OK")
import vipe
print("vipe: OK")
PY

echo
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Download reconstruction checkpoints with:"
echo "  bash scripts/models/download_reconstruction_checkpoints.sh required"
