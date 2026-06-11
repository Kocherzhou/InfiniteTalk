# 便携 worker 镜像 —— 任何平台几分钟拉起

目的：**不再被 AutoDL 一家的库存卡死**。镜像里固化了实测能跑的环境
（torch2.5.1/cu124/py3.12 + flash-attn + numpy2.0.2/scipy1.13.1/numba0.60）。
权重不在镜像里，运行时下到挂载卷。worker 只需**出站**连家里,任何能跑 GPU 的云都行。

适配 **Hopper/Ampere**：H100 / H200 / H800 / A100 / A800。
（Blackwell B200/PRO6000 需 cu128/torch2.7，另出 tag。）

## 1. 镜像怎么来
GitHub Actions 自动构建并推到 GHCR：`ghcr.io/kocherzhou/infinitetalk-worker:latest`
- 改了 Docker 相关文件会自动触发；也可在仓库 **Actions → build-worker-image → Run workflow** 手动触发。
- 构建成功后，到 GitHub **Packages** 把这个 package 设为 **Public**（一次性），各平台就能免登录 `docker pull`。

## 2. 在任意 GPU 云上跑（单卡）
```bash
docker pull ghcr.io/kocherzhou/infinitetalk-worker:latest

docker run -d --gpus all --name italk-worker \
  -v /data/italk-weights:/app/weights \      # 持久卷：权重下一次，重启复用
  -e HOME_BASE_URL=https://e.tangake.com:18444 \
  -e WORKER_TOKEN=ace0f86faed71c8f699e8f559a37c65d \
  -e T5_CPU=0 \                              # 80G 卡设 0（T5 进显存，省系统内存防 OOM）；48G 卡用 1
  ghcr.io/kocherzhou/infinitetalk-worker:latest
```
首次启动会自动下载 ~90G 权重到 `/app/weights`（挂卷里），之后秒起。

## 3. 多卡（4 卡示例）
```bash
docker run -d --gpus all --name italk-worker \
  -v /data/italk-weights:/app/weights \
  -e HOME_BASE_URL=https://e.tangake.com:18444 \
  -e WORKER_TOKEN=ace0f86faed71c8f699e8f559a37c65d \
  -e NGPUS=4 \                               # torchrun + ulysses 序列并行（多卡自动用 t5_fsdp）
  ghcr.io/kocherzhou/infinitetalk-worker:latest
```

## 3.5 RunPod Secure Cloud 单卡试水（首选）
现货最足、唯一同时满足「免登录拉 GHCR + 真·持久网络卷 + 只需出站」。
1. 控制台 **Storage → Network Volume** 建一个 **100GB** 卷（$0.07/GB·月 ≈ $7/月），选一个有 A100/H100 现货的区域。
2. **Deploy** 一台 **A100-80G**（$1.39/h PCIe）或 H100，附上刚建的卷。
3. **Container Image** 填 `ghcr.io/kocherzhou/infinitetalk-worker:latest`（公开，免登录）。
4. **环境变量**：
   ```
   HOME_BASE_URL=https://e.tangake.com:18444
   WORKER_TOKEN=ace0f86faed71c8f699e8f559a37c65d
   T5_CPU=1                      # 48-80G 都先用 1 稳妥
   WEIGHTS_DIR=/workspace/weights # RunPod 卷默认挂 /workspace → 权重落持久卷
   ```
   （把网络卷的挂载路径保持默认 `/workspace` 即可；`WEIGHTS_DIR` 让 entrypoint 自动把 `/app/weights` 软链过去。）
5. 首启自动下 ~90G 权重到卷里（一次），之后重开秒起。

## 4. 平台备忘
- **境外（RunPod/Vast/Lambda）**：H100/H200 货多。下载权重慢的话 `-e HF_ENDPOINT=`（用官方源）。
- **国内（恒源云/矩池云/潞晨云等）**：默认 hf-mirror 即可；支付/网络对家里服务器都友好。
- worker 只需出站，不用开放任何入站端口。
- 关键环境变量：`HOME_BASE_URL` `WORKER_TOKEN`（必填）、`NGPUS`（多卡）、`T5_CPU`（80G 卡设 0）、`HF_ENDPOINT`。
