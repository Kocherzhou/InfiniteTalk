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
import re
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
# 多卡：NGPUS>1 时用 torchrun 起多进程 + xdit 序列并行(ulysses)。Wan-14B 40 头，
# NGPUS 须能整除 40（1/2/4/5/8 都行）。LoRA 是先合进模型再 FSDP 分片，兼容 8 步加速。
NGPUS = int(os.environ.get("NGPUS", "1"))

WORK_DIR = ROOT / "_worker"
WORK_DIR.mkdir(exist_ok=True)

# ── 生成参数（与 cloud_batch.py 一致，按云卡调） ──────────────────────────────
CKPT_DIR     = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR  = "weights/chinese-wav2vec2-base"
INFINITETALK = "weights/InfiniteTalk/single/infinitetalk.safetensors"
SIZE         = "infinitetalk-480"
MODE         = "streaming"
MOTION_FRAME = 9
# 全精度（我们没下 fp8 量化包）。48G 卡跑全精度没问题。想用 fp8 再下量化包并设 True。
USE_FP8      = False
QUANT_DIR    = "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
# FusionX 步数蒸馏 LoRA：把 40 步压到 8 步、CFG 降到 1（每步少跑无条件分支）→ 约快 8-10 倍
# （实测 31min/clip → ~4min/clip）。这是让全精度 Wan-14B 可用的关键。设 LORA_DIR="" 可关。
LORA_DIR     = "weights/lora/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
LORA_SCALE   = 1.0
SAMPLE_STEPS = 8       # 配 FusionX：8 步；不用 LoRA 时回 40
SAMPLE_SHIFT = 2.0     # FusionX 推荐
# 一段视频最多生成多少帧。默认 1000 帧≈40 秒，会把更长的音频截断！
# 整首歌(单图)或长片段必须调大,否则只出 40 秒。流式逐窗生成,显存不随之增长。
MAX_FRAME_NUM = 100000
TEXT_GUIDE   = 1.0     # CFG=1（配蒸馏 LoRA），也省显存
AUDIO_GUIDE  = 2.0
# TeaCache：跳过扩散步间的冗余计算，Wan 系实测 ~1.5-2x 提速、基本不掉质量。
# thresh 越大越激进越快、质量风险越高；0.2 是 README 推荐的保守值。设 False 关闭。
USE_TEACACHE   = True
TEACACHE_THRESH = 0.2
# DiT 常驻显存参数量上限：None=全常驻最快(需 >47G 可用)；OOM 再设整数；极限设 0。
# 实测 vGPU-48G(可用 47.37G)全常驻会 OOM，设 11000000000(≈22G 常驻)稳；80G 卡留空走 None。
_npp = os.environ.get("NUM_PERSISTENT_PARAM_IN_DIT", "").strip()
NUM_PERSISTENT_PARAM_IN_DIT = int(_npp) if _npp else None
# 关键速度开关：offload_model 默认 True 会每步把 DiT 卸到 CPU（给小显存卡用），
# 在 48G 卡上慢 5-10 倍（曾实测 46s/step）。关掉它让 DiT 常驻显存。
OFFLOAD_MODEL = False
# T5 文本编码器放 CPU（只在开头编码一次提示词），给显存腾出 ~11G，配合 offload=False 防 OOM。
# 但放 CPU 会占 ~11G 系统内存——80G 显存的卡(H800)上长跑整首会把 96G RAM 撑爆(rc=-9 OOM)！
# 这种卡设 T5_CPU=0 让 T5 进显存、腾出系统内存。48G 卡才需要 =1。
T5_CPU       = os.environ.get("T5_CPU", "1") == "1"
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
        cli.post(f"{HOME_BASE_URL}/api/worker/result/{job_id}", data={"error": error[:1500]})
        return
    with open(mp4_path, "rb") as f:
        cli.post(f"{HOME_BASE_URL}/api/worker/result/{job_id}",
                 files={"video": (f"{job_id}.mp4", f, "video/mp4")},
                 timeout=600.0)


def build_cmd(input_json, save_stem):
    multi = NGPUS > 1
    if multi:
        # torchrun 单机多进程；--standalone 自带 rendezvous
        cmd = ["torchrun", "--nproc_per_node", str(NGPUS), "--standalone",
               "generate_infinitetalk.py"]
    else:
        cmd = [sys.executable, "generate_infinitetalk.py"]
    cmd += [
        "--ckpt_dir", CKPT_DIR,
        "--wav2vec_dir", WAV2VEC_DIR,
        "--infinitetalk_dir", INFINITETALK,
        "--input_json", str(input_json),
        "--size", SIZE,
        "--sample_steps", str(SAMPLE_STEPS),
        "--max_frame_num", str(MAX_FRAME_NUM),
        "--mode", MODE,
        "--motion_frame", str(MOTION_FRAME),
        "--sample_shift", str(SAMPLE_SHIFT),
        "--sample_text_guide_scale", str(TEXT_GUIDE),
        "--sample_audio_guide_scale", str(AUDIO_GUIDE),
        "--save_file", str(save_stem),
    ]
    if LORA_DIR:
        cmd += ["--lora_dir", LORA_DIR, "--lora_scale", str(LORA_SCALE)]
    if USE_TEACACHE:
        cmd += ["--use_teacache", "--teacache_thresh", str(TEACACHE_THRESH)]
    if multi:
        # 多卡：FSDP 分片 DiT/T5 + ulysses 序列并行；offload 由代码自动关，t5 用 fsdp 不用 cpu
        cmd += ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(NGPUS)]
    else:
        cmd += ["--offload_model", str(OFFLOAD_MODEL)]   # False = DiT 常驻显存（单卡快很多）
        if T5_CPU:
            cmd += ["--t5_cpu"]
        if NUM_PERSISTENT_PARAM_IN_DIT is not None:
            cmd += ["--num_persistent_param_in_dit", str(NUM_PERSISTENT_PARAM_IN_DIT)]
    if USE_FP8:
        cmd += ["--quant", "fp8", "--quant_dir", QUANT_DIR]
    return cmd


_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _run_and_stream(cli, job_id, cmd, env):
    """跑生成子进程，逐行解析输出，实时回报细粒度进度。返回 (returncode, full_log)。

    解析 generate_infinitetalk.py 的 stdout/stderr（合并）里的标记：
      - 'Creating ... pipeline' / 'loading ... weights' / 'WanModel' → 加载模型
      - 'Generating video'                                          → 开始生成
      - tqdm 步进度条 'x/N'（采样阶段，配 it/s|%||）                → 采样 第k段窗口 x/N 步
      - 'decode' / 'vae'                                            → VAE 解码
    tqdm 用 \\r 原地刷新，所以按 \\r 和 \\n 同时切行。
    """
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    full, window, last_step, stage, last_post = [], 0, -1, "", 0.0

    def emit(msg, prog, log=None):
        progress(cli, job_id, status="generating", progress=prog, message=msg, log=log)

    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch not in ("\r", "\n"):
            buf += ch
            continue
        line, buf = buf.strip(), ""
        if not line:
            continue
        full.append(line)
        if len(full) > 400:
            del full[:150]
        low = line.lower()
        now = time.time()
        if stage in ("", "load") and (("creating" in low and "pipeline" in low)
                or ("loading" in low and "weight" in low) or "wanmodel" in low):
            if stage != "load":
                stage = "load"
                emit("加载模型中（读权重，约几分钟）…", 25, line)
        elif "generating video" in low:
            if stage != "gen":
                stage = "gen"
                emit("开始生成…", 30, line)
        elif stage in ("gen", "sample") and ("decode" in low or "vae" in low):
            if stage != "decode":
                stage = "decode"
                emit("VAE 解码中…", 90, line)
        elif stage in ("gen", "sample"):
            m = _STEP_RE.search(line)
            if m and ("it/s" in low or "%" in line or "|" in line):
                s, t = int(m.group(1)), int(m.group(2))
                if 0 < t <= 100:                       # tqdm 采样条（~7/8 步），排除大数字
                    if last_step < 0 or s < last_step:  # 步数回落 = 进入新窗口
                        window += 1
                    last_step = s
                    stage = "sample"
                    if now - last_post >= 4:
                        emit(f"采样中 · 第 {window} 段窗口 · {s}/{t} 步", min(88, 32 + window * 2))
                        last_post = now
    proc.wait()
    return proc.returncode, "\n".join(full)


def run_infinitetalk(cli, job, img_path, aud_path):
    """真生成：subprocess 跑 generate_infinitetalk.py → save_stem.mp4（流式回报进度）。"""
    job_id = job["job_id"]
    save_stem = WORK_DIR / job_id
    input_json = WORK_DIR / f"{job_id}.json"
    input_json.write_text(json.dumps({
        "prompt": job.get("prompt", ""),
        "cond_video": str(img_path),
        "cond_audio": {"person1": str(aud_path)},
    }, ensure_ascii=False), encoding="utf-8")

    progress(cli, job_id, status="generating", progress=20,
             message="启动生成进程…", log="开始 InfiniteTalk 推理")
    cmd = build_cmd(input_json, save_stem)
    # expandable_segments 整理显存碎片，避免全精度+LoRA 在 48G 卡上 VAE 解码时 OOM；
    # OMP_NUM_THREADS 给 AutoDL 镜像的非法默认值兜底；PYTHONUNBUFFERED 保证输出实时不卡缓冲。
    gen_env = {**os.environ,
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "OMP_NUM_THREADS": "8",
               "PYTHONUNBUFFERED": "1"}
    rc, log = _run_and_stream(cli, job_id, cmd, gen_env)
    out_mp4 = Path(f"{save_stem}.mp4")
    if rc != 0 or not out_mp4.exists():
        raise RuntimeError(f"generate_infinitetalk rc={rc}: {log[-1500:]}")
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


def run_static(cli, job, img_path, aud_path):
    """画外音/空镜段：不跑 InfiniteTalk，ffmpeg 把静态图配音频 + 极缓慢推近(Ken Burns)。
    纯宇宙空镜(无人脸)走这条路：秒级、不耗 GPU、不会因检测不到人脸报错。
    STATIC_ZOOM=0 可关漂移退回纯静帧。输出保持原图宽高比，拼接时 stitch 再统一画幅。"""
    job_id = job["job_id"]
    out_mp4 = WORK_DIR / f"{job_id}.mp4"
    progress(cli, job_id, status="generating", progress=40,
             message="画外音空镜段：静态画面+音频（缓慢漂移）…", log="static/voiceover 渲染")

    def _probe(path, args):
        return subprocess.run(["ffprobe", "-v", "error", *args, str(path)],
                              capture_output=True, text=True).stdout.strip()
    try:
        dur = float(_probe(aud_path, ["-show_entries", "format=duration",
                                      "-of", "default=nw=1:nk=1"]))
    except Exception:
        dur = 10.0
    try:
        w, h = map(int, _probe(img_path, ["-select_streams", "v:0", "-show_entries",
                                          "stream=width,height", "-of", "csv=p=0"]).split(","))
    except Exception:
        w, h = 768, 768
    s = min(1.0, 960.0 / max(w, h))                 # 长边 ≤960
    tw = (int(round(w * s)) // 2 * 2) or 2
    th = (int(round(h * s)) // 2 * 2) or 2
    frames = max(1, int(round(dur * 25)))
    if os.environ.get("STATIC_ZOOM", "1") == "1":
        # Ken Burns 运镜。关键两点(2026-06-14 改进,修"傻乎乎抖动"):
        # ① 8× 预放大再 zoompan:旧版只 2× → 每帧 x/y 裁切坐标整数取整、缩放慢时逐帧
        #    跳 ~1px = 肉眼抖。预放大到长边 ~3840 → 每帧位移变亚像素级 → 丝滑。
        # ② 运动模式池:旧版每段都 1.0→1.15 居中推近、单调且幅度小。改成 5 种运镜
        #    (推近/拉远/左右移/上下移/起手放大再横移),按 job_id 散列轮换(worker 的
        #    job 无 seg_index),空镜段之间就有节奏变化。on=输出帧序号,线性匀速。
        long_edge = max(tw, th)
        ss = max(2, int(round(3840.0 / max(1, long_edge))))   # 长边预放大到 ~3840
        bw, bh = tw * ss, th * ss
        N = frames
        amp = 0.28                                            # 推/拉幅度 1.0→1.28
        cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        mode = int(job_id[:8], 16) % 5
        if mode == 0:        # 推近
            z, x, y = f"1+{amp}*on/{N}", cx, cy
        elif mode == 1:      # 拉远(起手放大、缓收)
            z, x, y = f"{1 + amp:.2f}-{amp}*on/{N}", cx, cy
        elif mode == 2:      # 左→右平移(轻放大留出余量)
            z, x, y = "1.12", f"(iw-iw/zoom)*on/{N}", cy
        elif mode == 3:      # 上→下平移
            z, x, y = "1.12", cx, f"(ih-ih/zoom)*on/{N}"
        else:                # 起手放大 1.30 再横移(揭示式)
            z, x, y = "1.30", f"(iw-iw/zoom)*on/{N}", cy
        vf = (f"scale={bw}:{bh}:flags=bicubic,"
              f"zoompan=z='{z}':d=1:x='{x}':y='{y}':s={tw}x{th}:fps=25,"
              f"setsar=1,format=yuv420p")
    else:
        vf = f"scale={tw}:{th},setsar=1,fps=25,format=yuv420p"
    # loudnorm 对齐响度:InfiniteTalk 人物段输出 ≈-23 LUFS,空镜原始切片偏响 ~6dB,
    # 不归一会在切到空镜时"音量跳高"(2026-06-13 实测)。I=-23 与人物段一致。
    cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", "25", "-t", f"{dur:.3f}",
           "-i", str(img_path), "-i", str(aud_path), "-vf", vf,
           "-af", "loudnorm=I=-23:TP=-1.5:LRA=11",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"static render failed: {(proc.stderr or '')[-300:]}")
    return out_mp4


def handle(cli, job, mock):
    job_id = job["job_id"]
    img = WORK_DIR / f"{job_id}_in.{job.get('image_ext', 'png')}"
    aud = WORK_DIR / f"{job_id}_in.{job.get('audio_ext', 'wav')}"
    progress(cli, job_id, status="generating", progress=10, message="下载输入…")
    download(cli, job_id, "image", img)
    download(cli, job_id, "audio", aud)

    if mock:
        runner = run_mock
    elif job.get("static"):
        runner = run_static          # 画外音/空镜段:静态图+音频,不对口型
    else:
        runner = run_infinitetalk
    out_mp4 = runner(cli, job, img, aud)

    progress(cli, job_id, status="uploading", progress=95, message="回传成品…", log="生成完成，回传中")
    post_result(cli, job_id, mp4_path=out_mp4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="不调大模型，用 ffmpeg 合占位 mp4（无 GPU 测试）")
    ap.add_argument("--once", action="store_true", help="只处理一个任务后退出（调试用）")
    args = ap.parse_args()

    print(f"cloud_worker → {HOME_BASE_URL}  (mock={args.mock}, NGPUS={NGPUS})")
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
