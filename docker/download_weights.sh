#!/usr/bin/env bash
# 下载 InfiniteTalk 所需全部权重到 ./weights（~90G）。幂等：已下的会续传/跳过。
# 全部走 HuggingFace + hf_transfer 加速：
#   - 境外平台：设 HF_ENDPOINT=https://huggingface.co（官方源，几十 MB/s）。
#   - 国内平台：保持默认 HF_ENDPOINT=https://hf-mirror.com（镜像站，同样快）。
# hf_transfer 偶尔会"连接挂起但不退出"卡在某个大文件，所以每个下载都包一层
# timeout + 续传重试：卡死 900s 就杀掉，断点续传重来，直到成功。
set -euo pipefail
cd /app
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p weights

# dl <repo> [透传给 huggingface-cli 的参数...]：卡死 15 分钟自动杀掉续传重试。
dl() {
  local repo="$1"; shift
  until timeout 900 huggingface-cli download "$repo" "$@"; do
    echo "[download] $repo 超时/中断 → 5s 后断点续传重试..."; sleep 5
  done
}

echo "[download] Wan2.1-I2V-14B-480P（~77G，HF + hf_transfer，带超时重试）..."
dl Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P

echo "[download] chinese-wav2vec2-base ..."
dl TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base
dl TencentGameMate/chinese-wav2vec2-base model.safetensors \
    --revision refs/pr/1 --local-dir weights/chinese-wav2vec2-base

echo "[download] InfiniteTalk single ..."
dl MeiGen-AI/InfiniteTalk --local-dir weights/InfiniteTalk \
    --include "single/*" "configuration.json"

echo "[download] FusionX LoRA ..."
dl vrgamedevgirl84/Wan14BT2VFusioniX \
    FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors --local-dir weights/lora

echo "[download] 全部权重下载完成"
