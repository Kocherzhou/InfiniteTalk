#!/usr/bin/env bash
# 容器入口：首次自动下载权重到挂载卷，然后起 worker。
set -euo pipefail
cd /app

# 权重目录适配任意平台的挂载点：默认 /app/weights；若平台把持久卷挂在别处
# （RunPod 默认 /workspace、潞晨/AutoDL 各有约定），设 WEIGHTS_DIR 指过去即可，
# entrypoint 会把 /app/weights 软链到它 —— 代码仍按 weights/ 相对路径找权重，
# 实际落在持久卷里，关机不丢、重启秒起。
WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
if [ "$WEIGHTS_DIR" != "/app/weights" ]; then
  echo "[entrypoint] WEIGHTS_DIR=$WEIGHTS_DIR → 软链 /app/weights 过去（持久卷）"
  mkdir -p "$WEIGHTS_DIR"
  rm -rf /app/weights 2>/dev/null || true
  ln -sfn "$WEIGHTS_DIR" /app/weights
fi
mkdir -p /app/weights

# 完整性标志用「最后一个下载的文件」（FusionX LoRA）：它存在 ⇒ 前面全下完了。
# 旧版用 config.json 会误判——它下得早，模型本体还没下完就被当成「已就位」跳过。
WEIGHTS_MARK="weights/lora/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
if [ ! -f "$WEIGHTS_MARK" ]; then
  echo "[entrypoint] 权重缺失 → 下载到 ./weights（挂卷可持久化，下次跳过）"
  bash docker/download_weights.sh
else
  echo "[entrypoint] 权重已就位，跳过下载"
fi

echo "[entrypoint] 启动 worker：HOME_BASE_URL=${HOME_BASE_URL:-未设!} NGPUS=${NGPUS:-1} T5_CPU=${T5_CPU:-1}"
exec python cloud_worker.py "$@"
