#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_mv.py —— InfiniteTalk「歌曲 + 一张照片 → MV」一键脚本

它做两件事：
  1) 根据你给的【照片】【歌曲音频】【画面描述】自动生成 InfiniteTalk 需要的 input JSON
  2) 拼好带 MV 优化参数的运行命令并执行（也可以 --dry-run 只打印命令）

用法示例（单人演唱）：
  python mv/make_mv.py \
      --photo   mv/assets/singer.png \
      --audio   mv/assets/song.wav \
      --prompt  "A singer performing on a neon-lit stage, close-up, cinematic lighting" \
      --size    720 \
      --name    song01

只想先看会跑什么命令、不真正渲染：加 --dry-run
显存不够：加 --low-vram （等价 --num_persistent_param_in_dit 0），或 --quant int8

注意：真正渲染需要一台有 NVIDIA GPU 的机器，并已按 README 下载好 weights/ 下的模型权重。
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 权重默认路径（与 README 一致），可用环境变量覆盖
DEFAULTS = {
    "ckpt_dir": os.environ.get("IT_CKPT_DIR", "weights/Wan2.1-I2V-14B-480P"),
    "wav2vec_dir": os.environ.get("IT_WAV2VEC_DIR", "weights/chinese-wav2vec2-base"),
    "infinitetalk_single": os.environ.get(
        "IT_INFINITETALK_SINGLE", "weights/InfiniteTalk/single/infinitetalk.safetensors"
    ),
    "infinitetalk_multi": os.environ.get(
        "IT_INFINITETALK_MULTI", "weights/InfiniteTalk/multi/infinitetalk.safetensors"
    ),
}


def build_parser():
    p = argparse.ArgumentParser(
        description="InfiniteTalk 一键 MV 生成（歌曲 + 照片）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # —— 素材 ——
    p.add_argument("--photo", required=True, help="人物照片路径（正脸清晰、勿裁脸）")
    p.add_argument("--audio", required=True, help="歌曲音频路径（person1 / 单人或主唱）")
    p.add_argument("--audio2", default=None, help="第二条音轨（男女对唱时的 person2）")
    p.add_argument(
        "--prompt",
        default=None,
        help="画面/场景描述（英文效果最好）。与 --prompt-file 二选一",
    )
    p.add_argument("--prompt-file", default=None, help="从文件读取画面描述")

    # —— 输出 ——
    p.add_argument("--name", default="mymv", help="本次任务名（决定 JSON 与成片文件名）")
    p.add_argument(
        "--size",
        choices=["480", "720"],
        default="480",
        help="分辨率：480 或 720",
    )

    # —— MV 优化参数（已给好对 MV 友好的默认值）——
    p.add_argument("--steps", type=int, default=40, help="采样步数")
    p.add_argument(
        "--audio-cfg",
        type=float,
        default=4.0,
        help="音频引导强度（口型同步）。不用 LoRA 时 3-5 最佳，越大口型越准",
    )
    p.add_argument(
        "--text-cfg",
        type=float,
        default=5.0,
        help="文本引导强度。不用 LoRA 时建议 5",
    )
    p.add_argument(
        "--color-correction",
        type=float,
        default=1.0,
        help="颜色校正强度 0~1，缓解长视频(>1分钟)掉色/串色",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=3000,
        help="最大帧数（25fps 下约 120 秒）。整首歌请按歌长调大",
    )
    p.add_argument("--motion-frame", type=int, default=9, help="长视频驱动帧长")
    p.add_argument(
        "--mode",
        choices=["streaming", "clip"],
        default="streaming",
        help="整首歌等长视频必须用 streaming；clip 只出一小段",
    )

    # —— 性能 / 显存 ——
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="低显存模式（--num_persistent_param_in_dit 0）",
    )
    p.add_argument("--quant", choices=["int8", "fp8"], default=None, help="量化以省显存")
    p.add_argument("--use-teacache", action="store_true", help="启用 TeaCache 加速")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="多卡数量。>1 时用 torchrun 并开启 FSDP + ulysses",
    )

    # —— 路径覆盖 ——
    p.add_argument("--ckpt-dir", default=DEFAULTS["ckpt_dir"])
    p.add_argument("--wav2vec-dir", default=DEFAULTS["wav2vec_dir"])
    p.add_argument(
        "--infinitetalk-dir",
        default=None,
        help="不填则按是否对唱自动选 single/multi 权重",
    )

    p.add_argument("--dry-run", action="store_true", help="只打印命令，不真正运行")
    return p


def resolve_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if args.prompt:
        return args.prompt
    # 给一个中性的兜底描述
    return (
        "A person passionately singing, expressive face, cinematic lighting, "
        "close-up shot, professional music video."
    )


def build_input_json(args, prompt):
    cond_audio = {"person1": args.audio}
    data = {"prompt": prompt, "cond_video": args.photo}
    if args.audio2:
        cond_audio["person2"] = args.audio2
        data["audio_type"] = "para"  # 两条音轨并行（对唱）
    data["cond_audio"] = cond_audio

    os.makedirs(os.path.join(REPO_ROOT, "mv", "configs"), exist_ok=True)
    json_path = os.path.join("mv", "configs", f"{args.name}.json")
    with open(os.path.join(REPO_ROOT, json_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return json_path


def build_command(args, json_path):
    is_multi = args.audio2 is not None
    it_dir = args.infinitetalk_dir or (
        DEFAULTS["infinitetalk_multi"] if is_multi else DEFAULTS["infinitetalk_single"]
    )
    save_file = os.path.join("mv", "outputs", args.name)

    base = [
        "--ckpt_dir", args.ckpt_dir,
        "--wav2vec_dir", args.wav2vec_dir,
        "--infinitetalk_dir", it_dir,
        "--input_json", json_path,
        "--size", f"infinitetalk-{args.size}",
        "--sample_steps", str(args.steps),
        "--sample_audio_guide_scale", str(args.audio_cfg),
        "--sample_text_guide_scale", str(args.text_cfg),
        "--color_correction_strength", str(args.color_correction),
        "--max_frame_num", str(args.max_frames),
        "--motion_frame", str(args.motion_frame),
        "--mode", args.mode,
        "--base_seed", str(args.seed),
        "--save_file", save_file,
    ]
    if args.low_vram:
        base += ["--num_persistent_param_in_dit", "0"]
    if args.quant:
        base += ["--quant", args.quant]
    if args.use_teacache:
        base += ["--use_teacache"]

    if args.gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "--standalone",
            "generate_infinitetalk.py",
            "--dit_fsdp",
            "--t5_fsdp",
            f"--ulysses_size={args.gpus}",
        ] + base
    else:
        cmd = [sys.executable, "generate_infinitetalk.py"] + base
    return cmd


def main():
    args = build_parser().parse_args()

    # 基本素材检查
    for label, path in [("照片", args.photo), ("音频", args.audio)]:
        if not os.path.exists(os.path.join(REPO_ROOT, path)) and not os.path.exists(path):
            print(f"⚠️  找不到{label}文件：{path}", file=sys.stderr)

    prompt = resolve_prompt(args)
    json_path = build_input_json(args, prompt)
    cmd = build_command(args, json_path)

    print("=" * 60)
    print(f"✅ 已生成输入配置：{json_path}")
    print(f"   画面描述：{prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"   分辨率：{args.size}P | 模式：{args.mode} | 音频CFG：{args.audio_cfg}")
    print("=" * 60)
    print("将运行命令：\n  " + " ".join(cmd))
    print("=" * 60)

    if args.dry_run:
        print("（--dry-run：只打印，不执行）")
        return

    if not os.path.isdir(os.path.join(REPO_ROOT, "weights")):
        print(
            "⚠️  未发现 weights/ 模型目录，无法真正渲染。\n"
            "    请先在有 GPU 的机器上按 README 下载模型权重，或加 --dry-run 仅查看命令。",
            file=sys.stderr,
        )
        sys.exit(2)

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
