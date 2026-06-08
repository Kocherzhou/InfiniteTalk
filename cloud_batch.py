#!/usr/bin/env python3
"""
cloud_batch.py — batch-run InfiniteTalk over a folder of (image + audio) pairs.

Intended to run on a rented cloud GPU (e.g. AutoDL RTX 4090 24GB / A100), where
the 14B Wan model actually fits — the local RTX 3080 (10GB) OOMs even in the
--num_persistent_param_in_dit 0 mode.

Workflow (occasional, a few clips at a time):
  1. local: Gemma4 writes the scene prompt, your TTS (kokoro/edge) makes the
     audio, you pick the portrait image.
  2. drop per-clip files into ./inbox/ :
         <name>.png   (or .jpg)   driving portrait image
         <name>.wav   (or .mp3)   the voice/song audio
         <name>.txt   (optional)  the InfiniteTalk scene prompt; if missing,
                                   DEFAULT_PROMPT is used.
  3. run:  python cloud_batch.py
  4. results land in ./outbox/<name>.mp4
  5. download outbox/, composite into the MV locally.

Resumable: a clip whose outbox/<name>.mp4 already exists is skipped, so a
re-run only does the missing/failed ones.
"""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── CONFIG (tune to the rented card) ─────────────────────────────────────────
INBOX  = ROOT / "inbox"
OUTBOX = ROOT / "outbox"
TMPDIR = ROOT / "save_audio"          # generate_infinitetalk.py caches audio emb here

CKPT_DIR        = "weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR     = "weights/chinese-wav2vec2-base"
INFINITETALK    = "weights/InfiniteTalk/single/infinitetalk.safetensors"
SIZE            = "infinitetalk-480"
MODE            = "streaming"          # streaming = long video; clip = one chunk
MOTION_FRAME    = 9

# Speed / VRAM knobs. On a 24GB+ card you do NOT need the ultra-slow CPU-stream
# mode the 3080 was forced into. Start here and adjust if you hit OOM:
USE_FP8         = True                 # fp8 quant model (present in weights/), less VRAM + faster
QUANT_DIR       = "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
SAMPLE_STEPS    = 40                   # drop to 8 (FusionX LoRA) / 4 (lightx2v) if you add that LoRA
# None  -> keep all DiT params resident (fastest, needs the VRAM headroom of a 24GB+ card)
# 0     -> stream every param from CPU (slowest, last resort if OOM)
# <int> -> cap resident params (middle ground)
NUM_PERSISTENT_PARAM_IN_DIT = None

# Fallback prompt when a clip has no <name>.txt. Have Gemma4 write better ones.
DEFAULT_PROMPT = ("A person sings passionately into a microphone, expressive "
                  "facial movements and natural lip sync, warm studio lighting, "
                  "close-up performance shot.")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac")
# ─────────────────────────────────────────────────────────────────────────────


def find_clips():
    """Pair each audio file in inbox/ with a same-stem image (and optional .txt)."""
    clips = []
    for audio in sorted(INBOX.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        image = next((INBOX / f"{audio.stem}{e}" for e in IMAGE_EXTS
                      if (INBOX / f"{audio.stem}{e}").exists()), None)
        if image is None:
            print(f"  ⚠ 跳过 {audio.name}：找不到同名图片（{'/'.join(IMAGE_EXTS)}）")
            continue
        ptxt = INBOX / f"{audio.stem}.txt"
        prompt = ptxt.read_text(encoding="utf-8").strip() if ptxt.exists() else DEFAULT_PROMPT
        clips.append({"name": audio.stem, "image": image, "audio": audio, "prompt": prompt})
    return clips


def build_cmd(clip, json_path, save_file):
    cmd = [
        sys.executable, "generate_infinitetalk.py",
        "--ckpt_dir", CKPT_DIR,
        "--wav2vec_dir", WAV2VEC_DIR,
        "--infinitetalk_dir", INFINITETALK,
        "--input_json", str(json_path),
        "--size", SIZE,
        "--sample_steps", str(SAMPLE_STEPS),
        "--mode", MODE,
        "--motion_frame", str(MOTION_FRAME),
        "--save_file", str(save_file),
    ]
    if USE_FP8:
        cmd += ["--quant", "fp8", "--quant_dir", QUANT_DIR]
    if NUM_PERSISTENT_PARAM_IN_DIT is not None:
        cmd += ["--num_persistent_param_in_dit", str(NUM_PERSISTENT_PARAM_IN_DIT)]
    return cmd


def main():
    OUTBOX.mkdir(exist_ok=True)
    TMPDIR.mkdir(exist_ok=True)
    (ROOT / "_jobs").mkdir(exist_ok=True)

    if not INBOX.exists():
        INBOX.mkdir()
        print(f"已创建 {INBOX} —— 放入 <名字>.png + <名字>.wav(+可选 <名字>.txt)后再运行。")
        return

    clips = find_clips()
    if not clips:
        print(f"{INBOX} 里没有可处理的 clip。")
        return

    print(f"发现 {len(clips)} 个 clip。输出 → {OUTBOX}")
    done = skipped = failed = 0
    for i, c in enumerate(clips, 1):
        out_mp4 = OUTBOX / f"{c['name']}.mp4"
        if out_mp4.exists():
            print(f"[{i}/{len(clips)}] {c['name']}: 已存在，跳过 ✓")
            skipped += 1
            continue

        # generate_infinitetalk.py appends nothing? README uses --save_file <stem>
        # and produces <stem>.mp4, so pass the stem path (no extension).
        save_stem = OUTBOX / c["name"]
        job_json = ROOT / "_jobs" / f"{c['name']}.json"
        job_json.write_text(json.dumps({
            "prompt": c["prompt"],
            "cond_video": str(c["image"]),          # a still image is accepted here
            "cond_audio": {"person1": str(c["audio"])},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = build_cmd(c, job_json, save_stem)
        print(f"[{i}/{len(clips)}] {c['name']}: 生成中…")
        print("    " + " ".join(shlex.quote(x) for x in cmd))
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        dt = time.time() - t0

        if rc == 0 and out_mp4.exists():
            print(f"    ✓ 完成 {out_mp4.name}（{dt:.0f}s）")
            done += 1
        else:
            print(f"    ✗ 失败 rc={rc}（{dt:.0f}s）—— 下次重跑会自动重试此条")
            failed += 1

    print(f"\n汇总：成功 {done}，跳过 {skipped}，失败 {failed}。")


if __name__ == "__main__":
    main()
