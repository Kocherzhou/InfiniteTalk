#!/usr/bin/env python3
"""
cloud_worker.py — 拉取式云端 worker（跑在租的 GPU 上，例如 AutoDL 4090）。

出站连家里的 Web app（内网穿透公网 URL）领活，下载 图片+音频，跑 InfiniteTalk
生成对口型视频，回传成品。云机**无需对外暴露**，只需能出站访问 HOME_BASE_URL。

环境变量：
  HOME_BASE_URL   家里 app 的可达地址，如 https://e.tangake.com:18444 或 http://localhost:28600
  WORKER_TOKEN    与家里 .env 的 WORKER_TOKEN 一致
  POLL_INTERVAL   轮询秒数（默认 5）

用法：
  # 真生成（在 InfiniteTalk 的 torch 环境里，权重已就位）
  HOME_BASE_URL=https://e.tangake.com:18444 WORKER_TOKEN=xxx python cloud_worker.py
  # 无 GPU 本地跑通整条链路（ffmpeg 合静态画面+音频占位 mp4）
  HOME_BASE_URL=http://localhost:28600 WORKER_TOKEN=xxx python cloud_worker.py --mock

生成参数沿用 cloud_batch.py 的 fp8/省显存配置（见下方 CONFIG）。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent

HOME_BASE_URL = os.environ.get("HOME_BASE_URL", "http://localhost:28600").rstrip("/")
WORKER_TOKEN  = os.environ.get("WORKER_TOKEN", "")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))

WORK_DIR = ROOT / "_worker"
WORK_DIR.mkdir(exist_ok=True)

# ── 生成参数（与 cloud_batch.py 一致，按云卡调） ──────────────────────────────
CKPT_DIR     = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR  = "weights/chinese-wav2vec2-base"
INFINITETALK = "weights/InfiniteTalk/single/infinitetalk.safetensors"
SIZE         = "infinitetalk-480"
MODE         = "streaming"
MOTION_FRAME = 9
USE_FP8      = True
QUANT_DIR    = "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
SAMPLE_STEPS = 40
NUM_PERSISTENT_PARAM_IN_DIT = None   # 24GB+ 用 None(最快)；OOM 再设整数；极限设 0
# ──────────────────────────────────────────────────────────────────────────────

# httpx 客户端：trust_env=False 绕过 .env 的 socks 代理（家里/云端都避免误走代理）。
_HDRS = {"X-Worker-Token": WORKER_TOKEN} if WORKER_TOKEN else {}


def _client():
    return httpx.Client(trust_env=False, timeout=120.0, headers=_HDRS)


def claim(cli):
    r = cli.post(f"{HOME_BASE_URL}/api/worker/claim")
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


def download(cli, job_id, kind, dest):
    with cli.stream("GET", f"{HOME_BASE_URL}/api/worker/input/{job_id}/{kind}") as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def progress(cli, job_id, **kw):
    try:
        cli.post(f"{HOME_BASE_URL}/api/worker/progress/{job_id}", json=kw)
    except Exception:
        pass


def post_result(cli, job_id, mp4_path=None, error=None):
    if error:
        cli.post(f"{HOME_BASE_URL}/api/worker/result/{job_id}", data={"error": error[:300]})
        return
    with open(mp4_path, "rb") as f:
        cli.post(f"{HOME_BASE_URL}/api/worker/result/{job_id}",
                 files={"video": (f"{job_id}.mp4", f, "video/mp4")},
                 timeout=600.0)


def build_cmd(input_json, save_stem):
    cmd = [
        sys.executable, "generate_infinitetalk.py",
        "--ckpt_dir", CKPT_DIR,
        "--wav2vec_dir", WAV2VEC_DIR,
        "--infinitetalk_dir", INFINITETALK,
        "--input_json", str(input_json),
        "--size", SIZE,
        "--sample_steps", str(SAMPLE_STEPS),
        "--mode", MODE,
        "--motion_frame", str(MOTION_FRAME),
        "--save_file", str(save_stem),
    ]
    if USE_FP8:
        cmd += ["--quant", "fp8", "--quant_dir", QUANT_DIR]
    if NUM_PERSISTENT_PARAM_IN_DIT is not None:
        cmd += ["--num_persistent_param_in_dit", str(NUM_PERSISTENT_PARAM_IN_DIT)]
    return cmd


def run_infinitetalk(cli, job, img_path, aud_path):
    """真生成：subprocess 跑 generate_infinitetalk.py → save_stem.mp4。"""
    job_id = job["job_id"]
    save_stem = WORK_DIR / job_id
    input_json = WORK_DIR / f"{job_id}.json"
    input_json.write_text(json.dumps({
        "prompt": job.get("prompt", ""),
        "cond_video": str(img_path),
        "cond_audio": {"person1": str(aud_path)},
    }, ensure_ascii=False), encoding="utf-8")

    progress(cli, job_id, status="generating", progress=20,
             message="云端生成中（InfiniteTalk）…", log="开始 InfiniteTalk 推理")
    cmd = build_cmd(input_json, save_stem)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out_mp4 = Path(f"{save_stem}.mp4")
    if proc.returncode != 0 or not out_mp4.exists():
        tail = (proc.stderr or proc.stdout or "")[-300:]
        raise RuntimeError(f"generate_infinitetalk rc={proc.returncode}: {tail}")
    return out_mp4


def run_mock(cli, job, img_path, aud_path):
    """无 GPU 占位：ffmpeg 把静态图片 + 音频合成一段 mp4（也是降级输出）。"""
    job_id = job["job_id"]
    out_mp4 = WORK_DIR / f"{job_id}.mp4"
    progress(cli, job_id, status="generating", progress=40,
             message="（mock）合成占位视频…", log="mock 模式：ffmpeg 静态画面+音频")
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(img_path), "-i", str(aud_path),
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"ffmpeg mock failed: {(proc.stderr or '')[-200:]}")
    return out_mp4


def handle(cli, job, mock):
    job_id = job["job_id"]
    img = WORK_DIR / f"{job_id}_in.{job.get('image_ext', 'png')}"
    aud = WORK_DIR / f"{job_id}_in.{job.get('audio_ext', 'wav')}"
    progress(cli, job_id, status="generating", progress=10, message="下载输入…")
    download(cli, job_id, "image", img)
    download(cli, job_id, "audio", aud)

    runner = run_mock if mock else run_infinitetalk
    out_mp4 = runner(cli, job, img, aud)

    progress(cli, job_id, status="uploading", progress=95, message="回传成品…", log="生成完成，回传中")
    post_result(cli, job_id, mp4_path=out_mp4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="不调大模型，用 ffmpeg 合占位 mp4（无 GPU 测试）")
    ap.add_argument("--once", action="store_true", help="只处理一个任务后退出（调试用）")
    args = ap.parse_args()

    print(f"cloud_worker → {HOME_BASE_URL}  (mock={args.mock})")
    if not args.mock and not (ROOT / "generate_infinitetalk.py").exists():
        print("⚠ 找不到 generate_infinitetalk.py —— 请在 InfiniteTalk 仓库根目录运行（或加 --mock）")
    idle_notified = False
    with _client() as cli:
        while True:
            try:
                job = claim(cli)
            except Exception as e:
                print(f"claim 失败（家里 app 不可达？）：{e}")
                time.sleep(POLL_INTERVAL)
                continue
            if not job:
                if not idle_notified:
                    print("等待任务…")
                    idle_notified = True
                time.sleep(POLL_INTERVAL)
                continue
            idle_notified = False
            jid = job["job_id"]
            print(f"接单 {jid}：{job.get('prompt','')[:50]}")
            try:
                handle(cli, job, args.mock)
                print(f"✓ 完成 {jid}")
            except Exception as e:
                print(f"✗ 失败 {jid}：{e}")
                try:
                    post_result(cli, jid, error=str(e))
                except Exception:
                    pass
            if args.once:
                break


if __name__ == "__main__":
    main()
