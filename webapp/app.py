#!/usr/bin/env python3
"""
数字人 MTV 工作台 — 家里端 Flask app (v1: 图+音 → 对口型视频).

纯界面 + 任务队列 + 文件存储 + SSE。**不 import torch / 不加载任何大模型** —
生成由云端 cloud_worker.py 出站领活完成 (pull 模式)。详见 webapp 同级的计划文件。

设计/脚手架风格沿用 ../video-subtitle (app.py 的 _auth_gate / jobs / SSE / add_log)。
"""
import os
import re
import time
import uuid
import json
import array
import shutil
import threading
import subprocess
from pathlib import Path

from flask import (Flask, request, jsonify, Response, render_template,
                   render_template_string, send_file, make_response, redirect,
                   stream_with_context)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

AUTH_TOKEN   = os.environ.get("AUTH_TOKEN", "").strip()      # 人登录口令 (空=不鉴权)
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "").strip()    # worker API 口令 (空=不鉴权)
PORT         = int(os.environ.get("PORT", 28600))

DEFAULT_PROMPT = ("A person performs into a microphone with expressive facial "
                  "movements and natural lip sync, close-up shot, warm lighting.")

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")
AUDIO_EXTS = ("wav", "mp3", "m4a", "flac", "aac", "ogg")

FFMPEG  = shutil.which("ffmpeg")  or "/home/kocher/miniconda3/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/home/kocher/miniconda3/bin/ffprobe"

# 拼接交叉淡化：相邻机位切换处视频做 N 帧 xfade + 音频同步 acrossfade（=0 走老硬切）。
# 刀口本就吸附到最静的换气处，音频淡化落在近无声区间，听感几乎无损；代价=整片比原曲短 N/25×(段数-1) 秒。
XFADE_FRAMES = int(os.environ.get("XFADE_FRAMES", "10"))   # 0=硬切（保留旧行为）
XFADE_FPS    = 25

# 人声活动检测（纯能量启发式，无 torch）：识别低能量的前奏/间奏 → 自动注入「安静聆听」提示词。
# 偏保守（宁漏不误）：只可靠识别 安静/近无声 段；响亮的纯器乐间奏未必识别，但绝不会误伤正常歌唱段。
VAD_QUIET_FRAC = float(os.environ.get("VAD_QUIET_FRAC", "0.06"))  # 帧能量 < 全曲 P90×此值 = 静
VAD_INTRO_MIN  = float(os.environ.get("VAD_INTRO_MIN",  "1.6"))   # 段首连续静 ≥ 此秒数 = 有前奏/间奏
VAD_INST_RATIO = float(os.environ.get("VAD_INST_RATIO", "0.65"))  # 整段静占比 ≥ 此值 = 基本无人声

# 云端 worker 在线/失联 + 异构调度
WORKER_ONLINE_SEC = int(os.environ.get("WORKER_ONLINE_SEC", "25"))   # 多久没轮询 = 失联（UI 用）
WORKER_ACTIVE_SEC = int(os.environ.get("WORKER_ACTIVE_SEC", "600"))  # 多久内算「在场」（调度用，覆盖一次生成）
FAST_RATIO        = float(os.environ.get("FAST_RATIO", "0.6"))       # 速度 ≥ 最快×此值 = 快卡


# ── 多机位 MTV 辅助：切歌 + 拼接（都在家里端用 ffmpeg） ───────────────────────
def _audio_duration(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def _video_dims(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", str(path)],
                         capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def _quietest_moment(src, center, win=4.0):
    """在 center±win 秒内找能量最低的 0.3s 窗口中心（≈人声换气/句间空隙），返回绝对秒数。
    解码 16k 单声道 PCM 后纯 Python 算滑窗能量，失败时退回 center。"""
    start = max(0.0, center - win)
    r = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{2*win:.3f}",
                        "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True)
    pcm = array.array("h")
    pcm.frombytes(r.stdout[: len(r.stdout) // 2 * 2])
    if len(pcm) < 16000:
        return center
    hop, span = 800, 4800                     # 50ms 步进、300ms 窗口 @16kHz
    best_i, best_e = 0, float("inf")
    for i in range(0, len(pcm) - span, hop):
        e = sum(s * s for s in pcm[i:i + span])
        if e < best_e:
            best_e, best_i = e, i
    return start + (best_i + span / 2) / 16000.0


def smart_cut_points(src, n):
    """整曲 n 段的下刀点列表（含 0 和结尾，共 n+1 个）。
    刀口不机械等分：每个等分点在 ±4s 内吸附到最静处，避免切在字/长音中间。"""
    dur = _audio_duration(src)
    seg = dur / n
    cuts = [0.0]
    for i in range(1, n):
        c = _quietest_moment(src, i * seg)
        if c < cuts[-1] + seg * 0.4:          # 防回退/段过短，退回等分点
            c = i * seg
        cuts.append(c)
    cuts.append(dur)
    return cuts


def split_audio_at(src, cuts, prefix):
    """按给定刀口 cuts（含 0 和结尾）把 src 切成 len(cuts)-1 段 wav（16k 单声道，
    InfiniteTalk 友好）。返回各段路径。刀口由 smart_cut_points 算好，与 VAD 复用同一份。"""
    n = len(cuts) - 1
    paths = []
    for i in range(n):
        p = f"{prefix}_seg{i}.wav"
        cmd = [FFMPEG, "-y", "-ss", f"{cuts[i]:.3f}", "-i", str(src)]
        if i < n - 1:
            cmd += ["-t", f"{cuts[i+1] - cuts[i]:.3f}"]   # 最后一段不限长，吃到结尾
        cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", p]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(p):
            raise RuntimeError(f"切第 {i+1} 段失败: {(r.stderr or '')[-200:]}")
        paths.append(p)
    return paths


def vocal_activity_profile(src, cuts):
    """整曲解码一次（16k 单声道），按 50ms 帧算能量，阈值取全曲 P90×VAD_QUIET_FRAC。
    按 cuts 切片，每段返回 {leading_quiet, quiet_ratio}（秒 / 占比）。
    纯能量启发式：可靠识别 安静/近无声 的前奏/间奏；响亮纯器乐段未必识别（偏保守，宁漏不误）。"""
    n = len(cuts) - 1
    blank = [{"leading_quiet": 0.0, "quiet_ratio": 0.0} for _ in range(n)]
    try:
        r = subprocess.run([FFMPEG, "-v", "error", "-i", str(src),
                            "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                           capture_output=True)
        pcm = array.array("h")
        pcm.frombytes(r.stdout[: len(r.stdout) // 2 * 2])
    except Exception:
        return blank
    FRAME = 800                                   # 50ms @16k
    nf = len(pcm) // FRAME
    if nf < 4:
        return blank
    energies = [sum(s * s for s in pcm[i * FRAME:(i + 1) * FRAME]) / FRAME for i in range(nf)]
    p90 = sorted(energies)[min(nf - 1, int(nf * 0.90))]
    thr = max(1.0, p90 * VAD_QUIET_FRAC)
    out = []
    for s in range(n):
        f0, f1 = min(nf, int(cuts[s] / 0.05)), min(nf, int(cuts[s + 1] / 0.05))
        fr = energies[f0:f1] or [p90]
        lead = 0
        for e in fr:
            if e < thr:
                lead += 1
            else:
                break
        quiet = sum(1 for e in fr if e < thr)
        out.append({"leading_quiet": lead * 0.05, "quiet_ratio": quiet / len(fr)})
    return out


def _media_duration(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def _norm_v(i, W, H):
    """把第 i 路视频统一到 W×H / 25fps / yuv420p / sar1，xfade 要求各路完全一致。"""
    return (f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={XFADE_FPS},"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{i}]")


def _stitch_hardcut(clip_paths, out_path):
    """老路线：统一缩放+补边到第一段尺寸，concat 硬切（=机位瞬切）。"""
    W, H = _video_dims(clip_paths[0])
    inputs, filt = [], []
    for i, c in enumerate(clip_paths):
        inputs += ["-i", str(c)]
        filt.append(_norm_v(i, W, H))
    concat_in = "".join(f"[v{i}][{i}:a]" for i in range(len(clip_paths)))
    filt.append(f"{concat_in}concat=n={len(clip_paths)}:v=1:a=1[v][a]")
    cmd = [FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg 拼接失败: {(r.stderr or '')[-200:]}")


def _stitch_xfade(clip_paths, out_path, dur):
    """机位切换处做 dur 秒视频 xfade + 同步音频 acrossfade。
    视频与音频淡化等长 → 整体等量缩短、A/V 始终同步（每段视频时长≈其音频时长）。
    刀口已吸附最静换气处，音频交叉淡化落在近无声区间，听感几乎无损。"""
    W, H = _video_dims(clip_paths[0])
    durs = [_media_duration(c) for c in clip_paths]
    d = min([dur] + [x * 0.45 for x in durs])     # 淡化时长须 < 任一段一半，防越界
    if d <= 0.02:
        raise RuntimeError("片段过短，无法交叉淡化")
    inputs, filt = [], []
    for i, c in enumerate(clip_paths):
        inputs += ["-i", str(c)]
        filt.append(_norm_v(i, W, H))
        filt.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                    f"asetpts=PTS-STARTPTS[a{i}]")
    last = len(clip_paths) - 1
    prev, off = "v0", durs[0] - d                 # offset = 累计时长 - 已用淡化
    for i in range(1, len(clip_paths)):
        tag = "vout" if i == last else f"vx{i}"
        filt.append(f"[{prev}][v{i}]xfade=transition=fade:duration={d:.3f}:"
                    f"offset={off:.3f}[{tag}]")
        prev, off = tag, off + durs[i] - d
    prevA = "a0"
    for i in range(1, len(clip_paths)):
        tag = "aout" if i == last else f"ax{i}"
        filt.append(f"[{prevA}][a{i}]acrossfade=d={d:.3f}[{tag}]")
        prevA = tag
    cmd = [FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
           "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"xfade 拼接失败: {(r.stderr or '')[-300:]}")


def stitch_clips(clip_paths, out_path):
    """按序把多段 mp4 拼成一片。XFADE_FRAMES>0 走交叉淡化（失败自动回退硬切），=0 硬切。"""
    if not clip_paths:
        raise RuntimeError("没有可拼接的片段")
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], out_path)
        return
    if XFADE_FRAMES > 0:
        try:
            _stitch_xfade(clip_paths, out_path, XFADE_FRAMES / XFADE_FPS)
            return
        except Exception as e:
            print(f"⚠ xfade 拼接失败，回退硬切：{e}")
    _stitch_hardcut(clip_paths, out_path)

app = Flask(__name__)

jobs = {}                 # job_id -> dict
_lock = threading.RLock()  # 可重入:存盘时可在持锁状态下再取锁做快照

# ── 任务落盘持久化(家里重启不丢整轮)+ 被领超时重领 ───────────────────────────
STATE_FILE = BASE / "jobs_state.json"
# 被领走却长时间无进展(worker 崩/网络假死)→ 退回队列让别的卡重领。
# 阈值要 > 单段最长无进展间隔:加载模型阶段约 9 分钟只报一次,故默认 15 分钟。
STALE_CLAIM_SEC = int(os.environ.get("STALE_CLAIM_SEC", "900"))
_save_lock = threading.Lock()


def save_jobs():
    """原子写盘:jobs → STATE_FILE(temp+rename)。任务就几个,频繁调用也廉价。"""
    with _save_lock:
        try:
            with _lock:
                payload = json.dumps(jobs, ensure_ascii=False)
            tmp = STATE_FILE.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            print(f"⚠ save_jobs 失败: {e}")


def load_jobs():
    """启动时从盘恢复任务。未完成任务保留状态(云端 worker 回报会续上),
    并给它们的 _updated 续命,避免一启动就被超时重领;在拼接中的父任务重新触发拼接。"""
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠ load_jobs 失败(忽略,从空开始): {e}")
        return
    now = time.time()
    for jid, job in data.items():
        if job.get("status") not in ("completed", "error"):
            job["_updated"] = now
        jobs[jid] = job
    alive = sum(1 for j in jobs.values() if j.get("status") not in ("completed", "error"))
    print(f"♻ 已从盘恢复 {len(jobs)} 个任务(未完成 {alive} 个)")
    for jid, job in list(jobs.items()):
        if job.get("kind") == "mtv" and job.get("status") == "stitching":
            threading.Thread(target=_do_stitch, args=(jid,), daemon=True).start()


def _requeue_stale_loop():
    """看门狗:被领走却 STALE_CLAIM_SEC 内无进展的段 → 退回 queued 等重领。"""
    while True:
        time.sleep(30)
        now, changed = time.time(), False
        with _lock:
            for job in jobs.values():
                if job.get("status") in ("claimed", "generating", "uploading") \
                        and now - job.get("_updated", now) > STALE_CLAIM_SEC:
                    job["status"] = "queued"
                    job["progress"] = 0
                    job["message"] = "云端 worker 失联,已退回队列等待重领…"
                    add_log(job, f"被领后 {STALE_CLAIM_SEC}s 无进展,自动退回队列重领", "error")
                    changed = True
        if changed:
            save_jobs()


# ── helpers ──────────────────────────────────────────────────────────────────
def _now():
    return time.strftime("%H:%M:%S")


def add_log(job, msg, log_type="info"):
    job.setdefault("logs", []).append({"time": _now(), "msg": msg, "type": log_type})
    job["logs"] = job["logs"][-50:]


def public_job(job):
    """Strip private (path) fields before sending to the browser."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _ext(filename, exts, default):
    e = filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    return e if e in exts else default


# ── 云端 worker 心跳：每次 /api/worker/* 接触即记 last-seen（失联标记 + 异构调度） ──
# worker 暂未上报身份/速度时，按来源 IP 区分、速度记为 None（隧道后多 worker 同 IP 会并成一个，
# 属已知局限，UI 仍能正确显示「在线/失联」）；待 worker 加 X-Worker-Id / X-Worker-Speed 后自动生效。
_workers = {}                       # worker_id -> {"last": ts, "speed": float|None}
_workers_lock = threading.Lock()


def _touch_worker(speed=None):
    wid = request.headers.get("X-Worker-Id") or request.remote_addr or "unknown"
    with _workers_lock:
        w = _workers.get(wid) or {}
        w["last"] = time.time()
        if speed is not None:
            w["speed"] = speed
        _workers[wid] = w
    return wid


def _fleet_snapshot():
    """返回 (online, active, last_ts)：online=近 WORKER_ONLINE_SEC 在轮询，active=近 WORKER_ACTIVE_SEC 在场。"""
    now = time.time()
    with _workers_lock:
        online = [(k, v) for k, v in _workers.items() if now - v["last"] < WORKER_ONLINE_SEC]
        active = [(k, v) for k, v in _workers.items() if now - v["last"] < WORKER_ACTIVE_SEC]
        last = max((v["last"] for v in _workers.values()), default=0.0)
    return online, active, last


# ── auth gate (clone of video-subtitle app.py:_auth_gate, + worker token) ──────
@app.before_request
def _auth_gate():
    path = request.path
    # Worker pull API: its own token, bypasses the human cookie login.
    if path.startswith("/api/worker/"):
        if WORKER_TOKEN and request.headers.get("X-Worker-Token", "") != WORKER_TOKEN:
            return jsonify({"error": "bad worker token"}), 401
        sp = request.headers.get("X-Worker-Speed")          # 心跳：记 last-seen(+速度)
        try:
            _touch_worker(float(sp) if sp else None)
        except (TypeError, ValueError):
            _touch_worker(None)
        return
    if not AUTH_TOKEN:
        return
    if request.endpoint == "login" or path == "/login":
        return
    tok = (request.args.get("token") or request.headers.get("X-Auth-Token")
           or request.cookies.get("auth_token"))
    if tok and tok == AUTH_TOKEN:
        if request.args.get("token"):
            resp = make_response(redirect(path))
            resp.set_cookie("auth_token", tok, max_age=30 * 24 * 3600,
                            httponly=True, samesite="Lax")
            return resp
        return
    if path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return redirect("/login")


LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>登录 · 数字人工作台</title><style>
body{background:#0d0d0d;color:#e2e2e2;font-family:system-ui,sans-serif;display:flex;
min-height:100vh;align-items:center;justify-content:center;margin:0}
.box{background:#161616;border:1px solid #2a2a2a;border-radius:14px;padding:32px;width:300px}
h1{font-size:17px;margin:0 0 18px;font-weight:600}
input{width:100%;padding:11px 12px;background:#1e1e1e;border:1px solid #333;color:#e2e2e2;
border-radius:8px;font-size:14px;box-sizing:border-box;outline:none}
button{width:100%;padding:11px;margin-top:14px;background:#38bdf8;border:0;color:#06283a;
border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.err{color:#f87171;margin-top:10px;font-size:13px;min-height:18px;text-align:center}
</style></head><body><form class="box" method="post">
<h1>🎤 数字人 MTV 工作台</h1>
<input type="password" name="password" placeholder="访问口令" autofocus>
<button type="submit">进入</button>
<div class="err">{{ err }}</div></form></body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_TOKEN:
        return redirect("/")
    if request.method == "POST":
        if request.form.get("password", "") == AUTH_TOKEN:
            resp = make_response(redirect("/"))
            resp.set_cookie("auth_token", AUTH_TOKEN, max_age=30 * 24 * 3600,
                            httponly=True, samesite="Lax")
            return resp
        return render_template_string(LOGIN_HTML, err="口令错误")
    return render_template_string(LOGIN_HTML, err="")


# ── user routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/create", methods=["POST"])
def api_create():
    image = request.files.get("image")
    audio = request.files.get("audio")
    if not image or not image.filename:
        return jsonify({"error": "请上传立绘/人像图片"}), 400
    if not audio or not audio.filename:
        return jsonify({"error": "请上传音频"}), 400

    prompt = (request.form.get("prompt") or "").strip() or DEFAULT_PROMPT
    job_id = uuid.uuid4().hex
    iext = _ext(image.filename, IMAGE_EXTS, "png")
    aext = _ext(audio.filename, AUDIO_EXTS, "wav")
    ipath = UPLOAD_DIR / f"{job_id}_input.{iext}"
    apath = UPLOAD_DIR / f"{job_id}_audio.{aext}"
    image.save(str(ipath))
    audio.save(str(apath))

    job = {
        "id": job_id, "status": "queued", "progress": 0,
        "message": "已入队，等待云端 worker 接单（请开启云机）…",
        "prompt": prompt, "image_name": image.filename, "audio_name": audio.filename,
        "logs": [], "created": time.time(),
        "_image": str(ipath), "_audio": str(apath),
    }
    add_log(job, "已上传，进入队列")
    with _lock:
        jobs[job_id] = job
    save_jobs()
    return jsonify({"job_id": job_id})


@app.route("/api/create_mtv", methods=["POST"])
def api_create_mtv():
    """多机位 MTV：N 张图 + 1 段歌 → 等分切歌 → N 个子任务入队 → 全部完成后自动拼接。
    云端 worker 不感知父子关系，只当普通子任务逐个领走生成。"""
    images = [im for im in request.files.getlist("images") if im and im.filename]
    audio = request.files.get("audio")
    if len(images) < 2:
        return jsonify({"error": "多机位需要至少 2 张图片（1 张请用单段模式）"}), 400
    if not audio or not audio.filename:
        return jsonify({"error": "请上传音频"}), 400

    prompt = (request.form.get("prompt") or "").strip() or DEFAULT_PROMPT
    # 逐张分镜提示词（与 images 同序）。非空=「分镜词, 全局基底」拼接；空=直接用全局。
    # 容错：吃掉用户从对话里复制来的尾部「+ 基底」占位符。
    per_prompts = [re.sub(r"[+＋]\s*基底\s*$", "", p.strip()).rstrip(" ,") for p in request.form.getlist("prompts")]
    # 逐张「画外音/空镜」标志(与 images 同序,"1"=该段不对口型、走静态图+音频)
    static_flags = request.form.getlist("static")
    n = len(images)
    parent_id = uuid.uuid4().hex
    apath = UPLOAD_DIR / f"{parent_id}_audio.{_ext(audio.filename, AUDIO_EXTS, 'wav')}"
    audio.save(str(apath))

    try:
        cuts = smart_cut_points(str(apath), n)                  # 智能刀口（吸附最静换气处）
        vad = vocal_activity_profile(str(apath), cuts)          # 复用同一刀口做人声活动检测
        segs = split_audio_at(str(apath), cuts, str(UPLOAD_DIR / parent_id))
    except Exception as e:
        return jsonify({"error": f"切歌失败：{str(e)[:150]}"}), 500

    t0 = time.time()
    parent = {
        "id": parent_id, "kind": "mtv", "status": "mtv_running", "progress": 3,
        "message": f"已切成 {n} 段，{n} 个子任务排队中，等待云端逐段生成…",
        "prompt": prompt, "logs": [], "created": t0, "n": n, "children": [],
        "vad": [],                                   # 人声活动检测预警（前奏/间奏注入提示）
    }
    add_log(parent, f"切歌完成（{n} 段），开始排队")

    child_jobs = []
    for i, im in enumerate(images):
        cid = uuid.uuid4().hex
        ipath = UPLOAD_DIR / f"{cid}_input.{_ext(im.filename, IMAGE_EXTS, 'png')}"
        im.save(str(ipath))
        capath = UPLOAD_DIR / f"{cid}_audio.wav"
        os.replace(segs[i], str(capath))           # 段 wav 改名成子任务音频
        per = per_prompts[i] if i < len(per_prompts) else ""
        child_prompt = f"{per}, {prompt}" if per else prompt
        # 剪接配套：边界动作归零，让相邻段切点两侧动作幅度收敛，减少跳变感
        child_prompt += ", starting from a calm settled pose, gently settling back to a quiet rest pose at the end"
        # 人声活动检测：低能量的前奏/间奏段 → 注入「安静聆听」抑制表演欲（取代旧的「假设前奏在段1」硬编码）
        prof = vad[i] if i < len(vad) else None
        if prof and prof["quiet_ratio"] >= VAD_INST_RATIO:
            child_prompt += (", this passage is an instrumental section with little or no singing; "
                             "he stays quiet and natural with mouth mostly closed, gently feeling the "
                             "music instead of singing")
            note = f"🟡 第{i+1}段：整体低能量（疑似器乐/间奏 {int(prof['quiet_ratio']*100)}%），已注入「间奏不开口」"
            parent["vad"].append(note); add_log(parent, note)
        elif prof and prof["leading_quiet"] >= VAD_INTRO_MIN:
            sec = prof["leading_quiet"]
            child_prompt += (f", during the quiet instrumental opening (about {sec:.0f} seconds) he listens "
                             "calmly with mouth closed and subtle breathing, only begins singing when the "
                             "vocals enter, restrained natural expression")
            note = f"🟡 第{i+1}段：开头约 {sec:.1f}s 低能量（疑似前奏/间奏），已注入「等人声再开口」"
            parent["vad"].append(note); add_log(parent, note)
        is_static = i < len(static_flags) and static_flags[i] == "1"
        child_jobs.append({
            "id": cid, "status": "queued", "progress": 0, "message": "排队中…",
            "prompt": child_prompt, "image_name": im.filename, "audio_name": f"第{i+1}段",
            "logs": [], "created": t0 + i * 0.001,   # 保序：seg0 先被领
            "_image": str(ipath), "_audio": str(capath),
            "_parent": parent_id, "seg_index": i, "static": is_static,
        })
        parent["children"].append(cid)

    with _lock:
        jobs[parent_id] = parent
        for c in child_jobs:
            jobs[c["id"]] = c
    save_jobs()
    return jsonify({"job_id": parent_id})


def _finalize_parent(child):
    """子任务完成/失败后调用：聚合父任务进度；全部完成则触发拼接。"""
    pid = child.get("_parent")
    parent = jobs.get(pid) if pid else None
    if not parent:
        return
    kids = [jobs.get(c) for c in parent["children"]]
    done = [k for k in kids if k and k.get("status") == "completed"]
    failed = [k for k in kids if k and k.get("status") == "error"]
    n = parent["n"]
    parent["progress"] = 5 + int(85 * len(done) / n)
    parent["message"] = f"已完成 {len(done)}/{n} 段" + (f"（{len(failed)} 段失败）" if failed else "")
    if len(done) + len(failed) < n:
        return
    if failed:
        parent["status"] = "error"
        parent["message"] = f"有 {len(failed)}/{n} 段失败，未拼接；已完成的段可单独下载"
        add_log(parent, parent["message"], "error")
        return
    parent["status"] = "stitching"
    parent["progress"] = 92
    parent["message"] = "全部段完成，正在拼接整片…"
    add_log(parent, "开始拼接 N 段")
    threading.Thread(target=_do_stitch, args=(pid,), daemon=True).start()


def _do_stitch(pid):
    parent = jobs.get(pid)
    try:
        clips = [str(UPLOAD_DIR / f"{c}_out.mp4") for c in parent["children"]]
        stitch_clips(clips, UPLOAD_DIR / f"{pid}_out.mp4")
        parent["status"] = "completed"
        parent["progress"] = 100
        parent["message"] = "多机位 MTV 拼接完成 ✓"
        add_log(parent, "拼接完成，整片已生成", "success")
    except Exception as e:
        parent["status"] = "error"
        parent["message"] = f"拼接失败：{str(e)[:120]}（子片段可单独下载）"
        add_log(parent, parent["message"], "error")
    save_jobs()


@app.route("/api/status/<job_id>")
def api_status(job_id):
    def generate():
        for _ in range(3600):
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': '任务不存在', 'status': 'error'})}\n\n"
                break
            yield f"data: {json.dumps(public_job(job), ensure_ascii=False)}\n\n"
            if job.get("status") in ("completed", "error"):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/result/<job_id>")
def api_result(job_id):
    out = UPLOAD_DIR / f"{job_id}_out.mp4"
    if not out.exists():
        return jsonify({"error": "结果不存在"}), 404
    as_dl = request.args.get("download") == "1"
    return send_file(str(out), mimetype="video/mp4", as_attachment=as_dl,
                     download_name=f"digitalhuman_{job_id}.mp4")


@app.route("/api/fleet")
def api_fleet():
    """云机在线/失联 + 队列概览，供前端轮询显示（用户最常踩的坑=忘了开 worker，活在队列干等）。"""
    online, active, last = _fleet_snapshot()
    now = time.time()
    with _lock:
        queued  = sum(1 for j in jobs.values() if j["status"] == "queued")
        running = sum(1 for j in jobs.values() if j["status"] in ("claimed", "generating", "uploading"))
    return jsonify({
        "online": len(online), "active": len(active),
        "last_seen_ago": int(now - last) if last else None,
        "queued": queued, "running": running,
    })


# ── worker pull API (X-Worker-Token; exempt from cookie gate) ──────────────────
@app.route("/api/worker/claim", methods=["POST"])
def worker_claim():
    sp = request.headers.get("X-Worker-Speed")
    try:
        my_speed = float(sp) if sp else None
    except ValueError:
        my_speed = None
    with _lock:
        queued = [j for j in jobs.values() if j["status"] == "queued"]
        # 异构调度（best-effort）：剩余段数 ≤ 在场快卡数 时，把活留给快卡，别让慢卡领走拖墙钟。
        # 仅当本卡已知速度且明显偏慢、且确有快卡在场时才让它稍等；无速度上报时退化为纯 FIFO（行为不变）。
        if my_speed is not None and queued:
            _, active, _ = _fleet_snapshot()
            speeds = [v["speed"] for _, v in active if v.get("speed")]
            if speeds:
                best = max(speeds)
                fast = [s for s in speeds if s >= FAST_RATIO * best]
                if my_speed < FAST_RATIO * best and len(queued) <= len(fast):
                    return ("", 204)          # 慢卡稍等：尾段留给快卡
        for job in sorted(queued, key=lambda j: j["created"]):
            job["status"] = "claimed"
            job["progress"] = 5
            job["message"] = "云端已接单，准备中…"
            job["_updated"] = time.time()
            add_log(job, "云端 worker 已接单")
            save_jobs()
            return jsonify({
                "job_id": job["id"],
                "prompt": job["prompt"],
                "image_ext": Path(job["_image"]).suffix.lstrip("."),
                "audio_ext": Path(job["_audio"]).suffix.lstrip("."),
                "static": bool(job.get("static")),
            })
    return ("", 204)


@app.route("/api/worker/input/<job_id>/<kind>")
def worker_input(job_id, kind):
    job = jobs.get(job_id)
    if not job:
        return ("", 404)
    path = job.get("_image") if kind == "image" else job.get("_audio") if kind == "audio" else None
    if not path or not os.path.exists(path):
        return ("", 404)
    return send_file(path)


@app.route("/api/worker/progress/<job_id>", methods=["POST"])
def worker_progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return ("", 404)
    d = request.get_json(force=True, silent=True) or {}
    for k in ("status", "progress", "message"):
        if k in d and d[k] is not None:
            job[k] = d[k]
    job["_updated"] = time.time()
    if d.get("log"):
        add_log(job, str(d["log"]), d.get("log_type", "info"))
    # 子任务进度时，顺带刷新父任务的"第 k/N 段生成中"提示
    pid = job.get("_parent")
    parent = jobs.get(pid) if pid else None
    if parent and parent.get("status") == "mtv_running":
        parent["message"] = f"第 {job.get('seg_index', 0) + 1}/{parent['n']} 段云端生成中…"
    save_jobs()
    return jsonify({"ok": True})


@app.route("/api/worker/result/<job_id>", methods=["POST"])
def worker_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return ("", 404)
    err = request.args.get("error") or request.form.get("error")
    if err:
        job["status"] = "error"
        job["message"] = f"云端生成失败：{err}"
        add_log(job, job["message"], "error")
        _finalize_parent(job)
        save_jobs()
        return jsonify({"ok": True})
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "no video file"}), 400
    out = UPLOAD_DIR / f"{job_id}_out.mp4"
    f.save(str(out))
    job["status"] = "completed"
    job["progress"] = 100
    job["message"] = "生成完成 ✓"
    add_log(job, "成品已回传", "success")
    _finalize_parent(job)
    save_jobs()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"🎤 数字人 MTV 工作台 starting on http://0.0.0.0:{PORT}")
    print(f"   AUTH_TOKEN:   {'✓ 已设置' if AUTH_TOKEN else '✗ 未设置(无登录)'}")
    print(f"   WORKER_TOKEN: {'✓ 已设置' if WORKER_TOKEN else '✗ 未设置(worker 无鉴权)'}")
    load_jobs()                                  # 重启恢复:内存队列不再因重启清空
    threading.Thread(target=_requeue_stale_loop, daemon=True).start()  # 被领超时重领看门狗
    print(f"   持久化: {STATE_FILE.name} | 超时重领: {STALE_CLAIM_SEC}s")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
