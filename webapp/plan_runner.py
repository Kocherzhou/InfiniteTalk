#!/usr/bin/env python3
"""plan_runner.py — 结构分镜排期 独立子进程(脱离 flask,重启打不死)。
用法: python3 plan_runner.py <tid>
读 uploads/adv_<tid>_{0..}.png + uploads/adv_<tid>_meta.json({song_id, lyrics, audio})
→ Suno 对齐歌词切镜头槽 → vision_advisor.plan_shots 排期 → 覆盖度体检
→ 进度/结果写 uploads/adv_<tid>_plan_status.json(flash 的 status 路由只读它)。
逐图理解走 GPU+节流,排期长推理走 CPU(护卡)。"""
import json, os, re, sys, glob, subprocess, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import vision_advisor as va

UP = os.path.join(BASE, "uploads")
tid = sys.argv[1]
STATUS = os.path.join(UP, f"adv_{tid}_plan_status.json")
SUNO = os.environ.get("SUNO_API", "http://localhost:3000")
TARGET, MAXLEN = 20.0, 26.0


def write(d):
    tmp = STATUS + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, STATUS)


def audio_dur(f):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", f], capture_output=True, text=True).stdout.strip())


def build_slots(song_id, song_dur):
    words = json.loads(urllib.request.urlopen(
        f"{SUNO}/api/get_aligned_lyrics?song_id={song_id}", timeout=60).read())
    secs = []
    for w in words:
        for m in re.finditer(r"\[([^\]]+)\]", w.get("word", "")):
            secs.append([m.group(1).strip()[:24], float(w.get("start_s", 0))])

    def sung(t):
        s = re.sub(r"\[[^\]]+\]", "", t); s = re.sub(r"[\(\)（）\s\n]", "", s); return len(s) > 0
    first = next((float(w["start_s"]) for w in words if sung(w.get("word", ""))), 0.0)
    last = max(float(w.get("end_s", 0)) for w in words)
    wstarts = sorted({float(w["start_s"]) for w in words if sung(w.get("word", ""))})
    ranges = []
    for i, (name, t) in enumerate(secs):
        end = secs[i + 1][1] if i + 1 < len(secs) else last
        if end - t < 1.0:
            continue
        ranges.append((name, max(t, first), end))
    if ranges:
        ranges[0] = (ranges[0][0], first, ranges[0][2])

    def snap(t):
        return min(wstarts, key=lambda x: abs(x - t)) if wstarts else t
    slots = []
    if first > 4:
        slots.append({"start": 0.0, "end": round(first, 2), "section": "前奏(器乐)", "role": "空镜"})
    for name, s, e in ranges:
        dur = e - s
        k = max(1, round(dur / TARGET)) if dur > MAXLEN else 1
        if k == 1:
            slots.append({"start": round(s, 2), "end": round(e, 2), "section": name, "role": "auto"})
        else:
            pts = sorted(set([round(s, 2)] + [round(snap(s + dur * j / k), 2) for j in range(1, k)] + [round(e, 2)]))
            for j in range(len(pts) - 1):
                slots.append({"start": pts[j], "end": pts[j + 1], "section": f"{name} ({j+1}/{k})", "role": "auto"})
    if song_dur - last > 4:
        slots.append({"start": round(last, 2), "end": round(song_dur, 2), "section": "尾奏(器乐)", "role": "空镜"})
    if slots:
        slots[-1]["end"] = round(song_dur, 2)
    for i, sl in enumerate(slots):
        sl["i"] = i; sl["dur"] = round(sl["end"] - sl["start"], 1)
    return slots


def coverage(slots, plan, descs):
    use = {}
    for p in plan:
        use[p.get("img")] = use.get(p.get("img"), 0) + 1
    n = len(descs)
    empties = [i + 1 for i, d in enumerate(descs) if ("无人" in d or "空镜" in d)]
    empty_slots = sum(1 for s in slots if s.get("role") == "空镜")
    tips = []
    unused = [i + 1 for i in range(n) if (i + 1) not in use]
    if unused:
        tips.append(f"有图没用上:{unused}")
    over = [k for k, v in use.items() if v >= 4]
    if over:
        tips.append(f"图{over}被用了4次+,可能审美疲劳(弹唱位除外)")
    if empty_slots > max(1, len(empties)):
        tips.append(f"空镜槽有{empty_slots}个、但真空镜图只有{len(empties)}张,建议补{empty_slots-len(empties)}张纯空镜(海浪/灯火/天空/乐器特写)")
    return {"usage": use, "empty_images": empties, "empty_slots": empty_slots,
            "tips": tips or ["配置健康,可直接上云机"]}


LIPSYNC_BUDGET = int(os.environ.get("LIPSYNC_BUDGET", "4"))   # 口型镜上限(其余空镜)


def pick_anchor(descs, plan):
    """弹唱位锚点:描述里在"对麦克风/唱歌/弹唱"的图;否则取 plan 里最常被指派的图。"""
    for i, d in enumerate(descs):
        if any(k in d for k in ("唱歌", "对麦", "麦克风", "弹唱")):
            return i + 1
    from collections import Counter
    c = Counter(p.get("img") for p in plan if isinstance(p, dict))
    return c.most_common(1)[0][0] if c else 1


def apply_lipsync_budget(slots, plan, descs, n_budget):
    """折衷:只 n_budget 个口型镜、且全锁定同一弹唱位锚点(绕开 InfiniTalk 跨场景不一致);
    其余全部空镜 KB(用军师指派的多样场景图,天然一致)。口型优先放副歌/华彩/桥段。"""
    anchor = pick_anchor(descs, plan)
    bys = {p["slot"]: p for p in plan if isinstance(p, dict)}
    sing = [s["i"] for s in slots if any(k in s["section"]
            for k in ("Chorus", "Bridge", "Final", "副歌", "华彩", "桥"))]
    pref = [i for i in sing if bys.get(i, {}).get("img") == anchor]    # 军师本就给了锚点的优先
    chosen = (pref + [i for i in sing if i not in pref])[:n_budget]
    for p in plan:
        if not isinstance(p, dict):
            continue
        if p["slot"] in chosen:
            p["img"] = anchor; p["static"] = False; p["role"] = "口型"
        else:
            p["static"] = True; p["role"] = "空镜"
    return {"anchor": anchor, "budget": n_budget, "lipsync_slots": sorted(chosen)}


def main():
    meta = json.load(open(os.path.join(UP, f"adv_{tid}_meta.json"), encoding="utf-8"))
    paths = sorted(glob.glob(os.path.join(UP, f"adv_{tid}_*.png")),
                   key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1)))
    n = len(paths)
    st = {"state": "running", "stage": "slots", "i": 0, "n": n, "msg": "切镜头槽…",
          "result": None, "error": None}
    write(st)
    song_dur = audio_dur(meta["audio"])
    slots = build_slots(meta["song_id"], song_dur)
    st["msg"] = f"切出 {len(slots)} 个镜头槽,开始逐图理解…"; write(st)

    def cb(stage, i, nn):
        st["stage"], st["i"], st["n"] = stage, i, nn
        if stage == "describe":
            t = va._gpu_temp()
            st["msg"] = f"逐图理解 {i+1}/{nn}" + (f"（GPU {t:.0f}°C节流）" if t else "")
        else:
            st["msg"] = f"分镜排期推演(CPU,{len(slots)}槽)…"
        write(st)

    out = va.plan_shots(paths, slots, meta.get("lyrics", ""), progress_cb=cb)
    # 折衷:口型预算 + 锁定弹唱位锚点,其余空镜
    budget = apply_lipsync_budget(slots, out["plan"], out["descriptions"], LIPSYNC_BUDGET)
    cov = coverage(slots, out["plan"], out["descriptions"])
    cov["tips"] = [f"折衷版:{budget['budget']} 口型(全用弹唱位图{budget['anchor']})+ "
                   f"{len(slots)-len(budget['lipsync_slots'])} 空镜,绕开跨场景不一致"] + cov["tips"]
    st.update(state="done", msg="分镜排期完成 ✓",
              result={"slots": slots, "plan": out["plan"], "budget": budget,
                      "descriptions": out["descriptions"], "coverage": cov})
    write(st)


try:
    main()
except Exception as e:
    json.dump({"state": "error", "error": str(e)[:400], "msg": "排期失败"},
              open(STATUS, "w", encoding="utf-8"), ensure_ascii=False)
