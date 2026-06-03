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
