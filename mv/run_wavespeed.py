#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_wavespeed.py —— 用「托管 API」跑 InfiniteTalk（不用自己装显卡）

这是「两手准备」里规模化那一手的过渡方案：同一个 InfiniteTalk 模型，
跑在 WaveSpeedAI 的云端，你只要给一张照片 URL + 一段音频 URL。

⚠️ 说明（重要）：
  - 本脚本仅依赖 Python 标准库（urllib），无需 pip 安装。
  - 端点(ENDPOINT)和字段名以 WaveSpeedAI 官方文档为准，可能随版本调整；
    本脚本未在本机联网验证，首次使用请对照官方文档校对一次。
  - 需要先准备好可公网访问的图片/音频 URL（多数平台要求传 URL 而非本地文件）。
  - 需要设置环境变量 WAVESPEED_API_KEY。

用法：
  export WAVESPEED_API_KEY=sk-xxxx
  python mv/run_wavespeed.py \
      --image https://your.cdn/singer.png \
      --audio https://your.cdn/song.wav \
      --out   mv/outputs/song01.mp4
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# 以官方文档为准。这里给出 WaveSpeedAI 的常见路径形态，便于你对照修改。
SUBMIT_ENDPOINT = os.environ.get(
    "WAVESPEED_ENDPOINT",
    "https://api.wavespeed.ai/api/v3/wavespeed-ai/infinitetalk",
)
RESULT_ENDPOINT = os.environ.get(
    "WAVESPEED_RESULT_ENDPOINT",
    "https://api.wavespeed.ai/api/v3/predictions/{id}/result",
)


def _post(url, payload, api_key):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, api_key):
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="WaveSpeedAI 托管版 InfiniteTalk")
    ap.add_argument("--image", required=True, help="人物照片的公网 URL")
    ap.add_argument("--audio", required=True, help="歌曲音频的公网 URL")
    ap.add_argument("--prompt", default="", help="可选画面描述")
    ap.add_argument("--resolution", default="720p", help="分辨率，如 480p / 720p")
    ap.add_argument("--out", default="mv/outputs/wavespeed_result.mp4")
    ap.add_argument("--poll", type=float, default=5.0, help="轮询间隔秒")
    args = ap.parse_args()

    api_key = os.environ.get("WAVESPEED_API_KEY")
    if not api_key:
        print("❌ 请先 export WAVESPEED_API_KEY=你的key", file=sys.stderr)
        sys.exit(1)

    # 字段名以官方文档为准；常见为 image / audio / prompt / resolution
    payload = {
        "image": args.image,
        "audio": args.audio,
        "prompt": args.prompt,
        "resolution": args.resolution,
    }

    print(f"➡️  提交任务到 {SUBMIT_ENDPOINT}")
    try:
        resp = _post(SUBMIT_ENDPOINT, payload, api_key)
    except urllib.error.HTTPError as e:
        print(f"❌ 提交失败 HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}", file=sys.stderr)
        print("   请对照 WaveSpeedAI 官方文档校对 ENDPOINT 与字段名。", file=sys.stderr)
        sys.exit(2)

    # 返回结构以官方为准；常见为 {"data": {"id": ...}}
    pred_id = (resp.get("data") or resp).get("id")
    if not pred_id:
        print(f"⚠️ 未拿到任务 id，原始返回：{json.dumps(resp, ensure_ascii=False)}")
        sys.exit(3)
    print(f"🆔 任务 id: {pred_id}，开始轮询结果……")

    result_url = RESULT_ENDPOINT.format(id=pred_id)
    video_url = None
    while True:
        time.sleep(args.poll)
        r = _get(result_url, api_key)
        d = r.get("data") or r
        status = d.get("status")
        print(f"   状态: {status}")
        if status in ("completed", "succeeded", "success"):
            outs = d.get("outputs") or d.get("output") or []
            video_url = outs[0] if isinstance(outs, list) and outs else outs
            break
        if status in ("failed", "error"):
            print(f"❌ 任务失败：{json.dumps(d, ensure_ascii=False)}", file=sys.stderr)
            sys.exit(4)

    if not video_url:
        print("⚠️ 任务完成但未取到视频 URL，请检查返回结构。", file=sys.stderr)
        sys.exit(5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"⬇️  下载成片到 {args.out}")
    urllib.request.urlretrieve(video_url, args.out)
    print(f"✅ 完成：{args.out}")


if __name__ == "__main__":
    main()
