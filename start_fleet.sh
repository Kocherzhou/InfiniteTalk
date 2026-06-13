#!/bin/bash
# start_fleet.sh — 单机多卡 worker 舰队。每张 GPU 起 1 个 cloud_worker.py(各钉一卡),
# 段任务从家里队列「贪心并行」领取:墙钟÷N、$/成片不变。
#
# ⚠️ 禁用 ulysses 合跑:实测 4 卡 ulysses 仅 2 倍速、通信税吃一半、$/成片翻倍。
#    一卡一 worker(本脚本)才是 $/产出最优。
# 中间文件按 job_id 命名空间隔离(save_audio/{job_id}_in/、_worker/{job_id}.mp4),
# 多 worker 同机共享目录零冲突。
#
# 用法(在 InfiniteTalk 仓库根、权重已就位):bash start_fleet.sh
#   看日志: tail -f _worker/worker_gpu0.log     停止: pkill -f cloud_worker.py
unset http_proxy https_proxy
export PYTHONUNBUFFERED=1   # worker 顶层 print 不缓冲,_worker/worker_gpuN.log 实时可读
export HOME_BASE_URL="${HOME_BASE_URL:-https://e.tangake.com:18444}"
export WORKER_TOKEN="${WORKER_TOKEN:-ace0f86faed71c8f699e8f559a37c65d}"
cd "$(dirname "$0")"

N=$(nvidia-smi -L | wc -l)
[ "${N:-0}" -ge 1 ] || { echo "没检测到 GPU"; exit 1; }

# 认卡(沿用 start_worker.sh 逻辑):cc>=10 走 bw(SDPA),老卡走 base(flash_attn)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | cut -d. -f1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
if [ "$CC" -ge 10 ]; then PYDIR=/root/miniconda3/envs/bw/bin; else PYDIR=/root/miniconda3/bin; fi
if [ "$MEM" -ge 70000 ]; then T5=0; NPP=""; else T5=1; NPP=11000000000; fi

echo "GPU×$N  cc=$CC  单卡=${MEM}MiB  python=$PYDIR  T5_CPU=$T5  NPP=${NPP:-none}"
mkdir -p _worker
for i in $(seq 0 $((N-1))); do
  echo "→ worker $i 绑 GPU $i,日志 _worker/worker_gpu$i.log"
  CUDA_VISIBLE_DEVICES=$i T5_CPU=$T5 NUM_PERSISTENT_PARAM_IN_DIT="$NPP" \
    nohup "$PYDIR/python" cloud_worker.py >> "_worker/worker_gpu$i.log" 2>&1 &
  # 错峰 90s:让 worker0 先把 89G 权重读进页缓存,后续 worker 从内存读(秒级),
  # 避免 4 卡同时冷读同一块系统盘 → 磁盘争抢把加载拖慢数倍(2026-06-13 实战教训)。
  [ "$i" -lt "$((N-1))" ] && sleep 90
done
echo "全部 $N 个 worker 已后台启动。pkill -f cloud_worker.py 可停。"
