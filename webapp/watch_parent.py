#!/usr/bin/env python3
import json, time, sys
from pathlib import Path
BASE = Path("/home/kocher/InfiniteTalk/webapp")
STATE = BASE / "jobs_state.json"
PID = "5d2c2be3bc4a4f7b9e436c05ec1aee4e"
for _ in range(180):  # 最多 ~90 分钟
    try:
        d = json.load(open(STATE))
        p = d.get(PID, {})
        st = p.get("status")
        kids = [d.get(c, {}) for c in p.get("children", [])]
        done = sum(1 for k in kids if k.get("status") == "completed")
        err = sum(1 for k in kids if k.get("status") == "error")
        n = p.get("n", len(kids))
    except Exception as e:
        print("read err:", e); time.sleep(20); continue
    if st == "completed":
        out = BASE / "uploads" / f"{PID}_out.mp4"
        print(f"DONE completed size={out.stat().st_size if out.exists() else 'NA'}")
        sys.exit(0)
    if st == "error":
        print(f"ERROR msg={p.get('message')} done={done} err={err}/{n}")
        sys.exit(0)
    print(f"[{time.strftime('%H:%M:%S')}] status={st} done={done} err={err} n={n}")
    time.sleep(30)
print("timeout")
