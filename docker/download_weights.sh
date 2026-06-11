#!/usr/bin/env bash
# 下载 InfiniteTalk 所需全部权重到 ./weights（~90G）。幂等：已下的会续传/跳过。
# 国内平台：默认 HF_ENDPOINT=hf-mirror（快）。境外平台：可 export HF_ENDPOINT= 用官方源。
set -euo pipefail
cd /app
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p weights

echo "[download] Wan2.1-I2V-14B-480P（~77G，走 ModelScope）..."
modelscope download --model Wan-AI/Wan2.1-I2V-14B-480P --local_dir weights/Wan2.1-I2V-14B-480P

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
