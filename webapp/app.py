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


def split_audio_even(src, n, prefix):
    """把 src 音频切成 n 段 wav（16k 单声道，InfiniteTalk 友好），刀口走 smart_cut_points。
    返回 n 个路径。"""
    cuts = smart_cut_points(src, n)
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


def stitch_clips(clip_paths, out_path):
    """按序把多段 mp4 拼成一片：统一缩放+补边到第一段的尺寸，硬切（=机位切换）。"""
    if not clip_paths:
        raise RuntimeError("没有可拼接的片段")
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], out_path)
        return
    W, H = _video_dims(clip_paths[0])
    inputs, filt = [], []
    for i, c in enumerate(clip_paths):
        inputs += ["-i", str(c)]
        filt.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v{i}]")
    concat_in = "".join(f"[v{i}][{i}:a]" for i in range(len(clip_paths)))
    filt.append(f"{concat_in}concat=n={len(clip_paths)}:v=1:a=1[v][a]")
    cmd = [FFMPEG, "-y", *inputs, "-filter_complex", ";".join(filt),
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg 拼接失败: {(r.stderr or '')[-200:]}")

app = Flask(__name__)

jobs = {}                 # job_id -> dict
_lock = threading.Lock()


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


# ── auth gate (clone of video-subtitle app.py:_auth_gate, + worker token) ──────
@app.before_request
def _auth_gate():
    path = request.path
    # Worker pull API: its own token, bypasses the human cookie login.
    if path.startswith("/api/worker/"):
        if WORKER_TOKEN and request.headers.get("X-Worker-Token", "") != WORKER_TOKEN:
            return jsonify({"error": "bad worker token"}), 401
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
    n = len(images)
    parent_id = uuid.uuid4().hex
    apath = UPLOAD_DIR / f"{parent_id}_audio.{_ext(audio.filename, AUDIO_EXTS, 'wav')}"
    audio.save(str(apath))

    try:
        segs = split_audio_even(str(apath), n, str(UPLOAD_DIR / parent_id))
    except Exception as e:
        return jsonify({"error": f"切歌失败：{str(e)[:150]}"}), 500

    t0 = time.time()
    parent = {
        "id": parent_id, "kind": "mtv", "status": "mtv_running", "progress": 3,
        "message": f"已切成 {n} 段，{n} 个子任务排队中，等待云端逐段生成…",
        "prompt": prompt, "logs": [], "created": t0, "n": n, "children": [],
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
        child_jobs.append({
            "id": cid, "status": "queued", "progress": 0, "message": "排队中…",
            "prompt": child_prompt, "image_name": im.filename, "audio_name": f"第{i+1}段",
            "logs": [], "created": t0 + i * 0.001,   # 保序：seg0 先被领
            "_image": str(ipath), "_audio": str(capath),
            "_parent": parent_id, "seg_index": i,
        })
        parent["children"].append(cid)

    with _lock:
        jobs[parent_id] = parent
        for c in child_jobs:
            jobs[c["id"]] = c
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


# ── worker pull API (X-Worker-Token; exempt from cookie gate) ──────────────────
@app.route("/api/worker/claim", methods=["POST"])
def worker_claim():
    with _lock:
        for job in sorted(jobs.values(), key=lambda j: j["created"]):
            if job["status"] == "queued":
                job["status"] = "claimed"
                job["progress"] = 5
                job["message"] = "云端已接单，准备中…"
                add_log(job, "云端 worker 已接单")
                return jsonify({
                    "job_id": job["id"],
                    "prompt": job["prompt"],
                    "image_ext": Path(job["_image"]).suffix.lstrip("."),
                    "audio_ext": Path(job["_audio"]).suffix.lstrip("."),
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
    if d.get("log"):
        add_log(job, str(d["log"]), d.get("log_type", "info"))
    # 子任务进度时，顺带刷新父任务的"第 k/N 段生成中"提示
    pid = job.get("_parent")
    parent = jobs.get(pid) if pid else None
    if parent and parent.get("status") == "mtv_running":
        parent["message"] = f"第 {job.get('seg_index', 0) + 1}/{parent['n']} 段云端生成中…"
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
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"🎤 数字人 MTV 工作台 starting on http://0.0.0.0:{PORT}")
    print(f"   AUTH_TOKEN:   {'✓ 已设置' if AUTH_TOKEN else '✗ 未设置(无登录)'}")
    print(f"   WORKER_TOKEN: {'✓ 已设置' if WORKER_TOKEN else '✗ 未设置(worker 无鉴权)'}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
