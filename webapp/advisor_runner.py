#!/usr/bin/env python3
"""advisor_runner.py — 军师独立子进程(脱离 flask,服务重启打不死它)。
用法:python3 advisor_runner.py <tid>
读 uploads/adv_<tid>_{0..n}.png + uploads/adv_<tid>_lyrics.txt,
跑 vision_advisor.analyze_sequence(逐图 GPU+节流,长推理 CPU),
进度/结果实时写 uploads/adv_<tid>_status.json;flask 的 status 路由只读这个文件。
"""
import json, os, sys, glob, time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import vision_advisor as va

UP = os.path.join(BASE, "uploads")
tid = sys.argv[1]
STATUS = os.path.join(UP, f"adv_{tid}_status.json")


def write(d):
    tmp = STATUS + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, STATUS)


paths = sorted(glob.glob(os.path.join(UP, f"adv_{tid}_*.png")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
lyr_f = os.path.join(UP, f"adv_{tid}_lyrics.txt")
lyrics = open(lyr_f, encoding="utf-8").read().strip() if os.path.exists(lyr_f) else ""
n = len(paths)
state = {"state": "running", "stage": "describe", "i": 0, "n": n,
         "msg": "军师启动…", "result": None, "error": None}
write(state)


def cb(stage, i, nn):
    state["stage"], state["i"], state["n"] = stage, i, nn
    if stage == "describe":
        t = va._gpu_temp()
        state["msg"] = (f"逐图理解 {i+1}/{nn}"
                        + (f"（GPU {t:.0f}°C，降到 {va.COOL_TARGET:.0f}°C 再喂下一张）"
                           if t is not None else ""))
    else:
        state["msg"] = "综合推演：排序+运镜（走 CPU，护卡，约 1-3 分钟）…"
    write(state)


try:
    out = va.analyze_sequence(paths, lyrics, progress_cb=cb)
    state.update(state="done", msg="军师参谋完成 ✓", result=out)
    write(state)
except Exception as e:
    state.update(state="error", error=str(e)[:300], msg="军师失败")
    write(state)
finally:
    for p in paths:
        try:
            os.remove(p)
        except Exception:
            pass
    if os.path.exists(lyr_f):
        try:
            os.remove(lyr_f)
        except Exception:
            pass
