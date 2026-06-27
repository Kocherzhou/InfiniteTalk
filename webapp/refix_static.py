#!/usr/bin/env python3
# 本地重渲染所有空镜段(抗抖好版 run_static),覆盖 uploads/<id>_out.mp4。分镜不动。
import json, os, subprocess, sys
from pathlib import Path

BASE = Path("/home/kocher/InfiniteTalk/webapp")
UPLOAD = BASE / "uploads"
STATE = BASE / "jobs_state.json"
PARENT = "1abce4a68d0747779073ff3c0e849a8a"

d = json.load(open(STATE))
parent = d[PARENT]
children = parent["children"]
print(f"父任务 {PARENT} · {len(children)} 段")

def probe(path, args):
    return subprocess.run(["ffprobe","-v","error",*args,str(path)],
                          capture_output=True,text=True).stdout.strip()

def render_static(job_id, img_path, aud_path, out_mp4):
    try:
        dur = float(probe(aud_path,["-show_entries","format=duration","-of","default=nw=1:nk=1"]))
    except Exception:
        dur = 10.0
    try:
        w,h = map(int, probe(img_path,["-select_streams","v:0","-show_entries",
                                       "stream=width,height","-of","csv=p=0"]).split(","))
    except Exception:
        w,h = 768,768
    s = min(1.0, 960.0/max(w,h))
    tw = (int(round(w*s))//2*2) or 2
    th = (int(round(h*s))//2*2) or 2
    frames = max(1,int(round(dur*25)))
    long_edge = max(tw,th)
    ss = max(2,int(round(3840.0/max(1,long_edge))))
    bw,bh = tw*ss, th*ss
    N = frames
    amp = 0.16
    cx,cy = "iw/2-(iw/zoom/2)","ih/2-(ih/zoom/2)"
    mode = int(job_id[:8],16)%5
    if mode==0:   z,x,y = f"1+{amp}*on/{N}", cx, cy
    elif mode==1: z,x,y = f"{1+amp:.2f}-{amp}*on/{N}", cx, cy
    elif mode==2: z,x,y = "1.12", f"(iw-iw/zoom)*on/{N}", cy
    elif mode==3: z,x,y = "1.12", cx, f"(ih-ih/zoom)*on/{N}"
    else:         z,x,y = "1.30", f"(iw-iw/zoom)*on/{N}", cy
    vf = (f"scale={bw}:{bh}:flags=bicubic,"
          f"zoompan=z='{z}':d=1:x='{x}':y='{y}':s={tw*2}x{th*2}:fps=25,"
          f"gblur=sigma=0.5,scale={tw}:{th}:flags=lanczos,"
          f"setsar=1,format=yuv420p")
    cmd = ["ffmpeg","-y","-loop","1","-framerate","25","-t",f"{dur:.3f}",
           "-i",str(img_path),"-i",str(aud_path),"-vf",vf,
           "-af","loudnorm=I=-23:TP=-1.5:LRA=11",
           "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
           "-c:a","aac","-b:a","192k","-shortest",str(out_mp4)]
    p = subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or not Path(out_mp4).exists():
        raise RuntimeError(f"render failed: {(p.stderr or '')[-300:]}")
    return tw,th,mode,dur

n_static=0
for i,cid in enumerate(children):
    j = d.get(cid,{})
    if not j.get("static"):
        continue
    img = j.get("_image"); aud = j.get("_audio")
    if not img or not os.path.exists(img):
        print(f"  ✗ seg#{i} {cid[:8]} 输入图缺失: {img}"); sys.exit(2)
    if not aud or not os.path.exists(aud):
        print(f"  ✗ seg#{i} {cid[:8]} 输入音缺失: {aud}"); sys.exit(2)
    out = UPLOAD / f"{cid}_out.mp4"
    tw,th,mode,dur = render_static(cid,img,aud,out)
    n_static+=1
    print(f"  ✓ seg#{i} {cid[:8]} 重渲 {tw}x{th} mode={mode} {dur:.1f}s seg_index={j.get('seg_index')}")

print(f"完成:重渲 {n_static} 段空镜")
