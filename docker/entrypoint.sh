#!/usr/bin/env bash
# 容器入口：首次自动下载权重到挂载卷 ./weights，然后起 worker。
set -euo pipefail
cd /app

WEIGHTS_MARK="weights/Wan2.1-I2V-14B-480P/config.json"
if [ ! -f "$WEIGHTS_MARK" ]; then
  echo "[entrypoint] 权重缺失 → 下载到 ./weights（挂卷可持久化，下次跳过）"
  bash docker/download_weights.sh
else
  echo "[entrypoint] 权重已就位，跳过下载"
fi

echo "[entrypoint] 启动 worker：HOME_BASE_URL=${HOME_BASE_URL:-未设!} NGPUS=${NGPUS:-1} T5_CPU=${T5_CPU:-1}"
exec python cloud_worker.py "$@"
