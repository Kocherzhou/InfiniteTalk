# AutoDL 跑真 worker 操作手册（照抄即可）

数字人 MTV 工作台 v1：家里 Web 界面派活 → AutoDL 云 GPU 跑 InfiniteTalk 出脸 → 回传。
本手册只管「云端 worker」这一侧。架构：**云机出站连家里领活，云机无需对外暴露**。

---

## Part 0 — 出发前在家做（最关键，缺一不可）

1. **设密钥并启动 webapp**（家里 WSL，~/InfiniteTalk/webapp）：
   ```bash
   cd ~/InfiniteTalk/webapp
   cp .env.example .env        # 然后编辑 .env：
   #   AUTH_TOKEN=<给人用的登录口令，公网暴露务必设>
   #   WORKER_TOKEN=<给 worker 用的口令，随便一长串，记下来>
   #   PORT=28600
   bash start.sh
   ```
   检查：`curl -s -o /dev/null -w '%{http_code}\n' http://localhost:28600/login` → **200**。

2. **内网穿透**：在你的穿透侧（和 e.tangake.com:18443 同一套）把 **28600** 映射成一个**新公网端口**，
   例如 `e.tangake.com:18444`。记下**完整 URL**：`https://e.tangake.com:18444`（http 还是 https 看你穿透配置）。

3. **从外网验证穿透通**（用手机流量打开，别走家里 WiFi）：浏览器开 `https://e.tangake.com:18444/login`
   能看到登录页 = 通了。

4. **带走三样东西**：
   - 公网 URL（如 `https://e.tangake.com:18444`）
   - `WORKER_TOKEN` 的值
   - 仓库与分支：`https://github.com/Kocherzhou/InfiniteTalk.git` 分支 `claude/music-video-production-4RCPK`

---

## Part 1 — AutoDL 租机

- **镜像**：选 `PyTorch 2.x / Python 3.10 / CUDA 12.1`（与 README 的 torch 2.4.1+cu121 对齐最省事）。
- **GPU**：RTX 4090 24GB（够用）；想快/省心可选 A100。
- **数据盘**：权重约 60–80GB，必须放数据盘 `/root/autodl-tmp`（系统盘装不下）。
- 开机 → 进 **JupyterLab → 终端**。

---

## Part 2 — 拿代码 + 装环境（都装在大盘 autodl-tmp 上）

```bash
cd /root/autodl-tmp
git clone -b claude/music-video-production-4RCPK https://github.com/Kocherzhou/InfiniteTalk.git
cd InfiniteTalk

# 镜像若自带 torch 2.4.x+cu121 可跳过 torch/xformers 两行；否则按 README：
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install misaki[en] ninja psutil packaging wheel
pip install flash_attn==2.7.4.post1     # ← 最容易卡的一步，见下方排错
pip install -r requirements.txt
pip install httpx                        # worker 必需
```

检查：`python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True 4090`。

---

## Part 3 — 下权重到 ./weights（domestic 加速）

```bash
source /etc/network_turbo 2>/dev/null || export HF_ENDPOINT=https://hf-mirror.com   # AutoDL 学术加速 / 镜像
pip install -U "huggingface_hub[cli]"

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P  --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

检查：`du -sh weights/*` → Wan ≈28–32G、InfiniteTalk 十几~几十G、wav2vec 几百M。
**weights 必须在 InfiniteTalk/ 目录内**（worker 用相对路径 `weights/...`）。

---

## Part 4 — 先单机自测一条（确认这台机器能出脸，再接 worker）

```bash
python generate_infinitetalk.py \
  --ckpt_dir weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir weights/chinese-wav2vec2-base \
  --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
  --input_json examples/single_example_image.json \
  --size infinitetalk-480 --sample_steps 40 --mode streaming --motion_frame 9 \
  --quant fp8 --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
  --save_file /root/autodl-tmp/test_out
```

检查：几分钟后生成 `/root/autodl-tmp/test_out.mp4`。
- 若 **OOM**：命令尾加 `--num_persistent_param_in_dit 0`（更省显存、更慢）。

---

## Part 5 — 跑真 worker（连家里）

```bash
cd /root/autodl-tmp/InfiniteTalk
export HOME_BASE_URL="https://e.tangake.com:18444"   # ← 换成你的公网 URL
export WORKER_TOKEN="<和家里 .env 完全一致>"

# 先验连通（期望 200）：
curl -s -o /dev/null -w "home: %{http_code}\n" "$HOME_BASE_URL/login"

# 领一单就退，便于首测：
python cloud_worker.py --once
```

然后**在家里/手机打开 webapp**（`https://e.tangake.com:18444`，用 AUTH_TOKEN 登录）→ 上传立绘+音频+提示词 →
AutoDL 终端应打印 `接单 …→ ✓ 完成`，家里 UI 进度走完、可下载 mp4。

确认无误后**去掉 --once 常驻**（后台挂着）：
```bash
nohup python cloud_worker.py > worker.log 2>&1 &
tail -f worker.log        # 看日志
```

**用完关机**：AutoDL 控制台「关机」即停止计费；权重在 autodl-tmp 持久，下次开机直接用。
（更快：把装好的环境**存私有镜像**，下次开机免重装。）

---

## 排错清单

| 现象 | 原因 / 解法 |
|---|---|
| worker 打印「claim 失败/家里不可达」 | 家里 webapp 没起 / 穿透没通 / URL 或端口写错 / token 不符。先 `curl $HOME_BASE_URL/login` 验证 200。|
| `flash_attn` 装不上（最常见） | 编译慢/失败。优先选**自带 flash-attn 的镜像**；或去 flash-attn GitHub releases 找匹配 torch2.4-cu121-cp310 的预编译 whl 直接 `pip install <url>`。|
| generate rc≠0、日志含 OOM | 改 `cloud_worker.py` 顶部 `NUM_PERSISTENT_PARAM_IN_DIT = 0`，重跑。|
| 下权重极慢/失败 | 确认 `source /etc/network_turbo` 生效，或 `export HF_ENDPOINT=https://hf-mirror.com` 后重试（断点续传，重跑即可）。|
| worker 跑起来但家里 UI 不动 | token 不一致 → worker 收 401（worker 日志会显示 claim 失败）。核对两边 WORKER_TOKEN。|
| 想提速/省钱 | 下个 FusionX/lightx2v LoRA，`SAMPLE_STEPS` 从 40 → 8/4，提速约 5 倍（v2 再弄）。|

## 调参位置
`cloud_worker.py` 顶部 `CONFIG`：`USE_FP8 / SAMPLE_STEPS / NUM_PERSISTENT_PARAM_IN_DIT / SIZE` 等，改完重跑 worker。
