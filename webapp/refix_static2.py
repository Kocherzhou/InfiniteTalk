#!/usr/bin/env python3
# 重渲空镜:保留第一个空镜不动,其余强制用 {推近/左右移/拉远} 三种平滑运镜之一。
import json, os, subprocess, sys
from pathlib import Path

BASE = Path("/home/kocher/InfiniteTalk/webapp")
UPLOAD = BASE / "uploads"
STATE = BASE / "jobs_state.json"
PARENT = "5d2c2be3bc4a4f7b9e436c05ec1aee4e"

# 允许的运镜:0=平滑推近 1=拉远 2=左右移。按 seg#1/#3/#5/#7/#9 顺序指定。
FORCED_SEQ = [0, 1, 2, 0, 1]
AMP = 0.50        # 推/拉幅度:1.0↔1.50
PAN_ZOOM = 1.60   # 平移留白:横移距离更大

d = json.load(open(STATE))
parent = d[PARENT]
children = parent["children"]

def probe(path, args):
    return subprocess.run(["ffprobe","-v","error",*args,str(path)],
                          capture_output=True,text=True).stdout.strip()

def render(job_id, img, aud, out, mode):
    try: dur = float(probe(aud,["-show_entries","format=duration","-of","default=nw=1:nk=1"]))
    except Exception: dur = 10.0
    try: w,h = map(int, probe(img,["-select_streams","v:0","-show_entries","stream=width,height","-of","csv=p=0"]).split(","))
    except Exception: w,h = 768,768
    s = min(1.0, 960.0/max(w,h))
    tw = (int(round(w*s))//2*2) or 2
    th = (int(round(h*s))//2*2) or 2
    N = max(1,int(round(dur*25)))
    long_edge = max(tw,th)
    ss = max(2,int(round(3840.0/max(1,long_edge))))
    bw,bh = tw*ss, th*ss
    amp = AMP
    cx,cy = "iw/2-(iw/zoom/2)","ih/2-(ih/zoom/2)"
    if mode==0:    z,x,y = f"1+{amp}*on/{N}", cx, cy            # 平滑推近
    elif mode==1:  z,x,y = f"{1+amp:.2f}-{amp}*on/{N}", cx, cy  # 拉远
    else:          z,x,y = f"{PAN_ZOOM}", f"(iw-iw/zoom)*on/{N}", cy   # 左→右平移
    vf = (f"scale={bw}:{bh}:flags=bicubic,"
          f"zoompan=z='{z}':d=1:x='{x}':y='{y}':s={tw*2}x{th*2}:fps=25,"
          f"gblur=sigma=0.5,scale={tw}:{th}:flags=lanczos,"
          f"setsar=1,format=yuv420p")
    cmd = ["ffmpeg","-y","-loop","1","-framerate","25","-t",f"{dur:.3f}",
           "-i",str(img),"-i",str(aud),"-vf",vf,
           "-af","loudnorm=I=-23:TP=-1.5:LRA=11",
           "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
           "-c:a","aac","-b:a","192k","-shortest",str(out)]
    p = subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not Path(out).exists():
        raise RuntimeError(f"render failed: {(p.stderr or '')[-300:]}")
    return tw,th,dur

statics = [(i,c) for i,c in enumerate(children) if d.get(c,{}).get("static")]
print("空镜段:", [f"seg#{i}/{c[:8]}" for i,c in statics])
first = statics[0][1]
print(f"保留第一个空镜不动: seg#{statics[0][0]} {first[:8]}")

modes_name = {0:"推近",1:"拉远",2:"左右移"}
k = 0
for i,c in statics:
    if c == first:
        continue
    j = d[c]
    img,aud = j.get("_image"), j.get("_audio")
    if not (img and os.path.exists(img) and aud and os.path.exists(aud)):
        print(f"  ✗ seg#{i} {c[:8]} 输入缺失 img={img} aud={aud}"); sys.exit(2)
    mode = FORCED_SEQ[k % len(FORCED_SEQ)]; k += 1
    tw,th,dur = render(c, img, aud, UPLOAD/f"{c}_out.mp4", mode)
    print(f"  ✓ seg#{i} {c[:8]} 重渲 {tw}x{th} {dur:.1f}s 运镜={modes_name[mode]}")
print(f"完成:重渲 {k} 段空镜(第一个保留)")
