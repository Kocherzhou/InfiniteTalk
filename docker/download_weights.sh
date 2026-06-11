#!/usr/bin/env bash
# 下载 InfiniteTalk 所需全部权重到 ./weights（~90G）。幂等：已下的会续传/跳过。
# 全部走 HuggingFace + hf_transfer 加速：
#   - 境外平台：设 HF_ENDPOINT=https://huggingface.co（官方源，几十 MB/s）。
#   - 国内平台：保持默认 HF_ENDPOINT=https://hf-mirror.com（镜像站，同样快）。
# 注：旧版 Wan2.1 走 ModelScope，在境外机房只有 ~60kB/s（ETA 数十小时），已弃用。
set -euo pipefail
cd /app
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p weights

echo "[download] Wan2.1-I2V-14B-480P（~77G，HuggingFace + hf_transfer）..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P

echo "[download] chinese-wav2vec2-base ..."
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors \
    --revision refs/pr/1 --local-dir weights/chinese-wav2vec2-base

echo "[download] InfiniteTalk single ..."
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir weights/InfiniteTalk \
    --include "single/*" "configuration.json"

echo "[download] FusionX LoRA ..."
huggingface-cli download vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors --local-dir weights/lora

echo "[download] 全部权重下载完成"
