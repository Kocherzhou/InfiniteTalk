#!/usr/bin/env python3
"""
数字人 MTV 工作台 — 家里端 Flask app (v1: 图+音 → 对口型视频).

纯界面 + 任务队列 + 文件存储 + SSE。**不 import torch / 不加载任何大模型** —
生成由云端 cloud_worker.py 出站领活完成 (pull 模式)。详见 webapp 同级的计划文件。

设计/脚手架风格沿用 ../video-subtitle (app.py 的 _auth_gate / jobs / SSE / add_log)。
"""
import os
import time
import uuid
import json
import threading
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
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"🎤 数字人 MTV 工作台 starting on http://0.0.0.0:{PORT}")
    print(f"   AUTH_TOKEN:   {'✓ 已设置' if AUTH_TOKEN else '✗ 未设置(无登录)'}")
    print(f"   WORKER_TOKEN: {'✓ 已设置' if WORKER_TOKEN else '✗ 未设置(worker 无鉴权)'}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
