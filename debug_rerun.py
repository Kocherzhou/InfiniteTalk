#!/usr/bin/env python3
"""在 pod 上原样复跑最近一个任务,把真实报错直接打到终端。
用法(pod /app 下):
  curl -sL https://raw.githubusercontent.com/Kocherzhou/InfiniteTalk/claude/music-video-production-4RCPK/debug_rerun.py -o debug_rerun.py
  python debug_rerun.py            # 复跑最近的 _worker/*.json
  python debug_rerun.py 1          # 强制单卡跑(排除多卡因素)
"""
import glob, os, subprocess, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:                      # 允许临时覆盖 NGPUS
    os.environ["NGPUS"] = sys.argv[1]

from cloud_worker import build_cmd, NGPUS  # noqa: E402  复用 worker 的命令拼装

jsons = sorted(glob.glob("_worker/*.json"), key=os.path.getmtime)
if not jsons:
    sys.exit("`_worker/` 里没有任务 json —— 先在工作台提交过任务才有输入可复跑")
j = jsons[-1]
cmd = build_cmd(j, "_worker/rerun_debug")
print(f"[rerun] NGPUS={NGPUS} input={j}\n[rerun] cmd: {' '.join(map(str, cmd))}\n", flush=True)
sys.exit(subprocess.run(list(map(str, cmd))).returncode)
