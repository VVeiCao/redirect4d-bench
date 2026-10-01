#!/bin/bash
# FreeOrbit4D 模型下载脚本
#
# 用法:
#   bash scripts/download_checkpoints.sh          # 下载全部必需模型
#   bash scripts/download_checkpoints.sh sv4d     # 只下载 SV4D  (Stage 0, ~24GB)
#   bash scripts/download_checkpoints.sh dpg      # 只下载 DPG   (Stage 1, ~6.4GB)
#   bash scripts/download_checkpoints.sh wan      # 只下载 Wan2.2 (Stage 2, ~76GB)
#   bash scripts/download_checkpoints.sh vggt     # 预下载 VGGT  (Stage 1, ~4GB, HF cache)
#   bash scripts/download_checkpoints.sh qwen     # 预下载 Qwen3-VL (Stage 2 caption, ~4GB, HF cache)
#   bash scripts/download_checkpoints.sh sam2     # 只下载 SAM2  (交互式标注, ~900MB)
#   bash scripts/download_checkpoints.sh all      # 下载全部

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CKPT_DIR="${PROJECT_ROOT}/checkpoints"

TARGET="${1:-required}"

echo "============================================================"
echo " FreeOrbit4D 模型下载"
echo "============================================================"
echo " 目标目录: ${CKPT_DIR}"
echo " 下载范围: ${TARGET}"
echo "============================================================"

# ---- SV4D (Stage 0): stabilityai/sv4d2.0 ----
download_sv4d() {
    echo -e "\n[SV4D] 下载 SV4D 多视角生成模型 (~12GB x2) ..."
    mkdir -p "${CKPT_DIR}/sv4d"

    if [ -f "${CKPT_DIR}/sv4d/sv4d2.safetensors" ]; then
        echo "  sv4d2.safetensors 已存在，跳过"
    else
        hf download stabilityai/sv4d2.0 sv4d2.safetensors \
            --local-dir "${CKPT_DIR}/sv4d"
    fi

    if [ -f "${CKPT_DIR}/sv4d/sv4d2_8views.safetensors" ]; then
        echo "  sv4d2_8views.safetensors 已存在，跳过"
    else
        hf download stabilityai/sv4d2.0 sv4d2_8views.safetensors \
            --local-dir "${CKPT_DIR}/sv4d"
    fi

    echo "[SV4D] 完成"
}

# ---- DPG (Stage 1 背景): PAGE-4D ----
download_dpg() {
    echo -e "\n[DPG] 下载 DPG 背景点云模型 (~6.4GB) ..."
    mkdir -p "${CKPT_DIR}/dpg"

    if [ -f "${CKPT_DIR}/dpg/checkpoint_150.pt" ]; then
        echo "  checkpoint_150.pt 已存在，跳过"
    else
        echo "  DPG 权重需要从 Google Drive 手动下载："
        echo ""
        echo "    https://drive.google.com/file/d/1c2G4z4sA3ouOmkPd2cZHDHnFB_n8LxU-/view"
        echo ""
        echo "  下载后放到: ${CKPT_DIR}/dpg/checkpoint_150.pt"
        echo ""
        # 尝试用 gdown 自动下载
        if command -v gdown &>/dev/null; then
            echo "  检测到 gdown，尝试自动下载 ..."
            gdown "1c2G4z4sA3ouOmkPd2cZHDHnFB_n8LxU-" -O "${CKPT_DIR}/dpg/checkpoint_150.pt"
        else
            echo "  提示: 安装 gdown 可自动下载: pip install gdown"
        fi
    fi

    echo "[DPG] 完成"
}

# ---- VGGT (Stage 1 前景): facebook/VGGT-1B ----
download_vggt() {
    echo -e "\n[VGGT] 预下载 VGGT-1B 前景点云模型 (~4GB) ..."
    echo "  模型会缓存到 HuggingFace cache 目录"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/VGGT-1B')
print('  VGGT-1B 下载完成')
"
    echo "[VGGT] 完成"
}

# ---- Wan2.2 (Stage 2): alibaba-pai/Wan2.2-VACE-Fun-A14B ----
download_wan() {
    echo -e "\n[Wan2.2] 下载 Wan2.2-VACE-Fun-A14B 视频生成模型 (~76GB) ..."
    WAN_DIR="${CKPT_DIR}/wan2.2"
    mkdir -p "${WAN_DIR}"

    # 主模型文件
    if [ -f "${WAN_DIR}/Wan2.1_VAE.pth" ]; then
        echo "  Wan2.2 模型已存在，跳过"
    else
        hf download alibaba-pai/Wan2.2-VACE-Fun-A14B \
            high_noise_model/diffusion_pytorch_model.safetensors \
            low_noise_model/diffusion_pytorch_model.safetensors \
            models_t5_umt5-xxl-enc-bf16.pth \
            Wan2.1_VAE.pth \
            --local-dir "${WAN_DIR}"
    fi

    # Tokenizer
    mkdir -p "${WAN_DIR}/tokenizer"
    if [ -d "${WAN_DIR}/tokenizer/umt5-xxl" ]; then
        echo "  Tokenizer 已存在，跳过"
    else
        hf download Wan-AI/Wan2.1-T2V-1.3B \
            google/umt5-xxl/tokenizer.json \
            google/umt5-xxl/tokenizer_config.json \
            --local-dir "${WAN_DIR}/tokenizer_tmp"
        mv "${WAN_DIR}/tokenizer_tmp/google/umt5-xxl" "${WAN_DIR}/tokenizer/umt5-xxl"
        rm -rf "${WAN_DIR}/tokenizer_tmp"
    fi

    echo "[Wan2.2] 完成"
}

# ---- Qwen3-VL (Stage 2 Caption, 可选): Qwen/Qwen3-VL-2B-Instruct ----
download_qwen() {
    echo -e "\n[Qwen3-VL] 预下载 Qwen3-VL-2B-Instruct Caption 模型 (~4GB) ..."
    echo "  模型会缓存到 HuggingFace cache 目录"
    hf download Qwen/Qwen3-VL-2B-Instruct
    echo "[Qwen3-VL] 完成"
}

# ---- SAM2 (交互式 Mask 标注): sam2_hiera_large.pt ----
download_sam2() {
    local sam2_dir="${CKPT_DIR}/sam2"
    local sam2_file="${sam2_dir}/sam2_hiera_large.pt"
    if [ -f "$sam2_file" ] || [ -L "$sam2_file" ]; then
        echo "[SAM2] 已存在，跳过: ${sam2_file}"
        return
    fi
    echo -e "\n[SAM2] 下载 SAM2 分割模型 (~900MB) ..."
    mkdir -p "$sam2_dir"
    wget -q --show-progress -O "$sam2_file" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    echo "[SAM2] 完成"
}

# ---- 按目标下载 ----
case "$TARGET" in
    sv4d)
        download_sv4d
        ;;
    dpg)
        download_dpg
        ;;
    vggt)
        download_vggt
        ;;
    wan)
        download_wan
        ;;
    qwen)
        download_qwen
        ;;
    sam2)
        download_sam2
        ;;
    required)
        download_sv4d
        download_dpg
        download_wan
        ;;
    all)
        download_sv4d
        download_dpg
        download_vggt
        download_wan
        download_qwen
        download_sam2
        ;;
    *)
        echo "用法: bash scripts/download_checkpoints.sh [sv4d|dpg|vggt|wan|required|all]"
        exit 1
        ;;
esac

echo -e "\n============================================================"
echo " 下载完成"
echo "============================================================"
echo ""
echo " checkpoints 目录结构:"
find "${CKPT_DIR}" -type f -o -type l | sort | while read f; do
    size=$(du -h "$f" 2>/dev/null | cut -f1)
    echo "   ${f#${PROJECT_ROOT}/}  (${size})"
done
echo ""
