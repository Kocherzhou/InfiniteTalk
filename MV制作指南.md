# 歌曲 → MV 制作指南

把已创作的歌曲做成 MV。核心思路：**一张人物照片 + 一段歌曲音频 → 人物跟着歌曲"开口演唱"的视频**（口型 / 表情 / 头部动作都与音乐对齐）。

本指南覆盖两条路线，建议**两手准备**：
- **路线一（成品平台）**：在线服务，零代码，先快速出片 → 推荐先用 **Kling Avatar 2.0** 试第一首。
- **路线二（自建 InfiniteTalk）**：用本仓库的开源模型，规模化、可控、不被平台绑定 → 见文末脚手架。

> 信息更新时间：2026-06。平台价格/限制会变，以官网为准。

---

## 一、路线对比速查

| | 方案 | 本质 | 单首歌成本(3-4分钟) | 上手 | 画质/可控 | 整首歌一次成型 |
|---|---|---|---|---|---|---|
| A | 自建 InfiniteTalk | 自己租 GPU 跑开源 | GPU 租金几毛~1刀 | ⚠️ 高 | 720P，**完全可控** | 需 streaming + 调 max_frame |
| B | InfiniteTalk 托管 API（WaveSpeedAI） | 同模型，云端跑 | 按量付费，便宜 | ✅ 低（调 API） | 720P，最长 10 分钟 | ✅ |
| C | **Kling Avatar 2.0** | 成品商业产品 | 标准≈$17 / Pro≈$35（5分钟） | ✅✅ 最低 | **1080P/48fps** | ✅ 单次 5 分钟 |
| C | Hedra（Character-3） | 成品，口型最强 | ≈$2.7/分钟 | ✅✅ 低 | 高，720P | ⚠️ 低档限 60 秒/次，长歌需分段 |

**关键技术点**：InfiniteTalk 单图生成**超过 1 分钟会掉色/串色**；而一首歌 3-4 分钟。
- 成品平台（Kling Avatar 2.0）单次可生成 5 分钟，**直接绕开这个坑**，最适合"整首歌 MV"。
- 自建时用 `--color_correction_strength`、或先把照片转成"轻微推拉的微动视频"再当输入来缓解。

---

## 二、路线一：成品平台（推荐先跑通第一首）

### 🥇 Kling Avatar 2.0（首选）
- **能力**：一张照片 + 一段音频，支持**唱歌/说唱**，单次最长 **5 分钟**、**1080P / 48fps**。
- **输入要求**：
  - 照片：正脸或 3/4 侧脸、眼睛可见、**脸别裁掉**；JPG / PNG / WebP，分辨率越高越好。
  - 音频：MP3 / WAV / M4A，**≤ 5 分钟**。
  - 可选：一句文字描述风格/镜头（英文最佳）。
- **网页版步骤（试第一首最快、零代码）**：
  1. 准备一张正脸清晰照片 + 歌曲音频（≤5 分钟）。
  2. 打开 klingai.com，选 **AI Avatar / Avatar 2.0**。
  3. 上传照片 + 上传歌曲音频。
  4. （可选）填一句场景描述（录音棚 / 舞台 / 灯光氛围）。
  5. 生成，得到 1080P 成片。
- **API 价格（规模化时）**：标准档 $0.056/秒（≈$17/5分钟）、Pro 档 $0.115/秒（≈$35/5分钟）。
  可走 fal.ai / kie.ai / WaveSpeedAI。

### 🥈 Hedra（Character-3）
- 口型同步业内最准，说/唱/rap 都行，无水印、可商用。
- **短板**：单次生成长度受限（免费 1 分钟，付费档放宽到 5 分钟），长歌可能要**分段拼接**。
- 价格：720P = 6 积分/秒（=360 积分/分钟）；Creator $30/月 = 4000 积分 ≈ 11 分钟成片（≈$2.7/分钟）。

### 🥉 InfiniteTalk on WaveSpeedAI（你选的模型的云端版）
- 一图 + 音频 → 最长 **10 分钟 / 720P**，REST API，便宜。
- 用本仓库脚本：`mv/run_wavespeed.py`（见下）。

---

## 三、路线二：自建 InfiniteTalk（本仓库脚手架）

> 需要一台有 NVIDIA GPU（建议 ≥24GB 显存）的机器，并按主 `README.md` 下载好 `weights/` 模型权重、装好 `requirements.txt` 依赖。
> 本仓库当前容器无 GPU、无权重，**只能生成命令/配置，不能真正渲染**。

### 目录结构
```
mv/
├── assets/      # 放你的歌曲音频 + 人物照片
├── configs/     # 自动生成的 input JSON
├── outputs/     # 成片输出（已 gitignore）
├── make_mv.py        # 一键：照片+歌曲+描述 → 生成JSON并运行
└── run_wavespeed.py  # 托管API版（不用显卡）
```

### 一键生成（单人演唱）
```bash
python mv/make_mv.py \
    --photo  mv/assets/singer.png \
    --audio  mv/assets/song.wav \
    --prompt "A singer performing on a neon-lit stage, cinematic close-up, warm lighting" \
    --size   720 \
    --name   song01
```
- 先看命令不渲染：加 `--dry-run`
- 显存不够：加 `--low-vram` 或 `--quant int8`
- 长视频掉色：调 `--color-correction 1.0`（默认已开）
- 整首歌：用默认 `--mode streaming`，并按歌长把 `--max-frames` 调大（25fps：帧数 ≈ 秒数 ×25）
- 口型更准：`--audio-cfg` 调到 4~5
- 多卡：`--gpus 8`（自动用 torchrun + FSDP）

### 男女对唱
```bash
python mv/make_mv.py \
    --photo mv/assets/duet.png \
    --audio mv/assets/vocal_man.wav \
    --audio2 mv/assets/vocal_woman.wav \
    --prompt "A man and a woman singing together, stage lighting" \
    --name duet01
```
（会自动写入 `audio_type: para` 并切换到 multi 权重。）

### 托管 API 版（不用自己装显卡）
```bash
export WAVESPEED_API_KEY=你的key
python mv/run_wavespeed.py \
    --image https://your.cdn/singer.png \
    --audio https://your.cdn/song.wav \
    --out   mv/outputs/song01.mp4
```
> 端点与字段名以 WaveSpeedAI 官方文档为准，首次使用请对照校对。

---

## 三·补充：低显存自建实战记录（10GB/RTX 3080）

> 在 RTX 3080 10GB + WSL2 上实测自建**原版** InfiniteTalk 的全过程结论。**核心教训：显存 ≤16GB 不要硬刚原版仓库，直接上 Wan2GP。**

### ✅ 最终推荐：低显存就用 Wan2GP（官方 Community Works 推荐）
[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) 专为"GPU-Poor"优化，6-12GB 显存即可跑 InfiniteTalk，把下面所有坑都自动堵上：
- **1-click 安装器**自动配齐加速内核（Triton / Sage / Flash / GGUF / **Lightx2v** / Nunchaku），免去手动版本地狱；
- 30 系吃 **Sage2 注意力**（最高 ~2× 提速）+ **int8 量化** + **Lightx2v 4 步蒸馏**（内置）；
- 智能 **VRAM Profile**（自动块交换），10GB 选"低显存档"；
- **Gradio 网页 UI**，反复做 MV 更顺手；
- 建议**原生 Windows 跑**（拿满系统内存，绕开 WSL 内存上限）。

### ⚠️ 原版仓库在 10GB 上的坑（已逐个踩过）
1. **模型装不下→搬运慢**：14B 模型 fp8 也要 ~14GB > 10GB 显存。用 `--num_persistent_param_in_dit 0` 全卸载 → 每次前向都从内存搬参数 → GPU 利用率仅 ~50%（传输瓶颈）、单步极慢。调高 `num_persistent`（如 5e9）可缓解但显存贴边（9960/10240MiB）易 OOM。
2. **fp8 量化的 CUDA 编译**：不带 `--t5_cpu` 时 optimum-quanto 要现编 marlin 内核，缺 `cuda_runtime_api.h`（无 CUDA 开发头文件）即失败。带 `--t5_cpu` 走 CPU 内核可绕过。
3. **bf16 吃内存**：不量化的 bf16 加载峰值 ~43-60GB；**WSL2 默认只给 ~31GB 内存** → `Killed`(OOM)。需在 Windows 的 `C:\Users\<你>\.wslconfig` 写 `[wsl2]\nmemory=50GB\nswap=32GB`，再 `wsl --shutdown` 重启生效。
4. **依赖版本三角**（亲测可用组合）：
   - `torch==2.4.1 (cu121)` + `xformers==0.0.28`
   - `transformers==4.49.0`（5.x 不兼容）
   - `diffusers==0.33.1`（0.38 的 attention_dispatch 与 torch2.4 冲突；0.31 又缺 xfuser 要的 `sana_transformer`；0.33.1 是甜点位）
   - `flash_attn==2.7.4.post1` 用预编译 wheel（`cu12torch2.4cxx11abiFALSE-cp310`），免 nvcc 编译
5. **t5_cpu 的 bug**（本仓库已修，commit `Fix t5_cpu context...`）：`wan/multitalk.py` 的 t5_cpu 分支把 `context` 多包了一层 list，导致 `multitalk_model.forward` 报 `'list' object has no attribute 'dtype'`。修法：该分支末尾对 context/context_null 取 `[0]`。
6. **权重下载**：新版 `hf`(huggingface_hub 1.17) 不认 `HF_ENDPOINT` 镜像、hf-xet 还绕过镜像 → 国内直接用 **ModelScope**：`modelscope download --model Wan-AI/Wan2.1-I2V-14B-480P` 和 `MeiGen-AI/InfiniteTalk`。
7. **散热**：offload 把 CPU（如 i9-11900K）烤到 94°C 降频；给 GPU 设 **70% 功率墙** + 用 Wan2GP 的高效负载可明显缓解。

### 何时仍用原版仓库
显存 **≥24GB**（RTX 3090/4090）：整模型直接进显存、零搬运、原生 CUDA，原版仓库 + `mv/make_mv.py` 跑得又快又稳。**这才是想本地爽跑的真正硬件解**（比上 Mac 实在——InfiniteTalk 是 CUDA 生态，Apple Metal 无官方支持）。

---

## 四、画面描述（prompt）怎么写

英文、具体、像在描述一个镜头。模板：
> `[人物] is passionately singing, [穿着], [场景/道具], [光线氛围], [镜头景别]. [情绪/风格].`

示例（官方）：
> "A woman is passionately singing into a professional microphone in a recording studio. She wears large black headphones and a dark cardigan. Warm, focused lighting. A close-up shot captures her expressive performance."

---

## 五、建议的推进节奏

1. **先用 Kling Avatar 2.0 网页版**出第一首，验证人物照片 + 歌曲的实际效果。
2. 满意后，决定规模化路线：
   - 量小/求省心 → 继续用 Kling（API 批量）。
   - 量大/要控制 → 用 `mv/run_wavespeed.py`（托管 InfiniteTalk）验证，再考虑自建压成本。

---

## 来源

- [Infinitalk API on fal.ai](https://fal.ai/models/fal-ai/infinitalk)
- [InfiniteTalk on WaveSpeedAI](https://wavespeed.ai/models/wavespeed-ai/infinitetalk)
- [Kling AI Avatar v2 Pro (fal.ai)](https://fal.ai/models/fal-ai/kling-video/ai-avatar/v2/pro)
- [Kling AI Avatar 2.0 指南 (kie.ai)](https://kie.ai/kling-ai-avatar)
- [Kling AI Launches Avatar 2.0 (Medium)](https://medium.com/@CherryZhouTech/kling-ai-launches-avatar-2-0-generate-expressive-5-minute-performances-from-a-single-image-11ba637472a6)
- [Hedra Pricing](https://www.hedra.com/pricing)
- [Best AI tools for singing avatars (Lemonslice)](https://lemonslice.com/blog/best-tools-singing-avatars)
- [fal.ai vs Replicate GPU pricing 2026](https://apidog.com/blog/best-ai-inference-platform-guide-2026/)
