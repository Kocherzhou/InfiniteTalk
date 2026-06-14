#!/usr/bin/env python3
"""vision_advisor.py — 用本地 Ollama gemma4 多模态对多机位排序做语义级预判。

两段式:
  1) describe_image: 逐图理解(景别/机位角度/人物姿态/构图/光线),结构化一句话。
  2) analyze_sequence: 把按序描述 + 歌曲情绪 + 剪辑原则喂给 gemma4(纯文本),
     产出首图/尾图评估、逐段衔接红黄绿+建议、整体观感预判、建议排序。

被 app.py 调用(也可 `python3 vision_advisor.py img1 img2 ...` 单测)。
gemma4 是 thinking 模型,务必 think=False(否则输出夹带推理)。
"""
import base64, json, os, re, subprocess, time, urllib.request

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma4:12b-it-qat")

# ── 温控节流(熬粥):每次推理前确保 GPU 已降到安全温度再喂,杜绝持续负载累积热死。
# 修散热(PTM7950)前也能用本地 3080 跑轻 GPU 活;修好后 GPU_THROTTLE=0 关掉、跑得更快。
COOL_TARGET = float(os.environ.get("COOL_TARGET", "55"))   # 降到此温度(°C)才喂下一块(实测55起跑峰值~65,留5°C余量)
COOL_CEIL   = float(os.environ.get("COOL_CEIL", "70"))     # 安全上限(仅告警)
COOL_MAX    = float(os.environ.get("COOL_MAX", "180"))     # 单次冷却最多等多久(秒)
THROTTLE    = os.environ.get("GPU_THROTTLE", "1") == "1"


def _gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _cooldown(tag=""):
    """喂下一块推理前,等 GPU 降到 COOL_TARGET 以下(熬粥式节流)。读不到温度则放行。"""
    if not THROTTLE:
        return
    t0 = time.time()
    while True:
        t = _gpu_temp()
        if t is None:
            return
        if t <= COOL_TARGET:
            print(f"  [温控] {tag} GPU {t:.0f}°C ≤ {COOL_TARGET:.0f},喂下一块")
            return
        if time.time() - t0 > COOL_MAX:
            print(f"  [温控] {tag} 等 {COOL_MAX:.0f}s 仍 {t:.0f}°C,放行(查散热/室温)")
            return
        print(f"  [温控] {tag} GPU {t:.0f}°C,等降到 {COOL_TARGET:.0f}…")
        time.sleep(8)


def _ollama(prompt, images=None, timeout=180, cpu=False):
    """cpu=True：强制 num_gpu=0 走 CPU(不碰显卡、无需冷却)。
    用于"最后一步长推理"——几十秒连续满载是没修的 3080 的死穴(节流插不进去),
    放 CPU 慢一点但绝对安全;逐图理解(短脉冲)仍走 GPU+节流。"""
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False, "think": False}
    if cpu:
        payload["options"] = {"num_gpu": 0}                # Ollama:0 层上 GPU = 纯 CPU
    else:
        _cooldown("推理前")                                # GPU 路径:喂前先确保已凉
    if images:
        payload["images"] = images
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if not cpu:
        tp = _gpu_temp()
        if tp is not None and tp > COOL_CEIL:
            print(f"  ⚠ [温控] 推理后 GPU {tp:.0f}°C(>{COOL_CEIL:.0f}),下次会多等会儿")
    return (r.get("response") or "").strip()


def describe_image(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    p = ("用中文简洁描述这张MV机位图,只客观描述不发挥,一行内涵盖:"
         "景别(特写/近景/中景/全景)、机位角度(正面/侧面/俯/仰/平视)、"
         "人物数量与姿态朝向、构图、光线明暗。")
    return _ollama(p, images=[b64])


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def analyze_sequence(image_paths, song_hint="", progress_cb=None):
    """返回 dict:{descriptions:[...], result:{first,last,transitions[],overall,suggested_order,motions[]}}

    motions[]:逐图运镜建议,每项 {img, motion, reason};motion ∈ 推近/拉远/左右移/上下移/放大横移。
    渲染端按 motion 选 KB 模式(语义化运镜),不再用 hash 随机。
    progress_cb(stage, i, n):可选进度回调,stage='describe'(逐图)/'reason'(推演)。"""
    n = len(image_paths)
    descriptions = []
    for i, p in enumerate(image_paths):
        if progress_cb:
            progress_cb("describe", i, n)
        descriptions.append(describe_image(p))
    if progress_cb:
        progress_cb("reason", n, n)
    listing = "\n".join(f"第{i+1}张: {d}" for i, d in enumerate(descriptions))
    hint = song_hint or "一首怀旧温暖的男女对唱(圆舞曲),情绪曲线:前奏安静→主歌收敛→副歌放开→结尾归于平静。"
    prompt = f"""你是资深MV剪辑指导。下面是一首歌按当前顺序排好的 {n} 个机位画面的描述。
歌曲信息(可能是完整歌词;请据歌词的叙事走向与情绪曲线——起/承/转/合、主歌收敛、副歌放开、桥段、结尾归静——来判断画面该怎么对位排列):
{hint}

机位描述(当前顺序):
{listing}

请结合"歌词叙事/情绪曲线"和"剪辑30度规则(相邻镜头要换景别或转30度以上机位,否则像跳切)"评估这个排序,只输出一个JSON对象,字段:
- "first": 首图是否适合开场(要有建立镜头/较大景别/安定感,且呼应歌词开头),一句话评价+建议
- "last": 尾图是否适合收尾(要有收束/归于平静感,且呼应歌词结尾),一句话评价+建议
- "transitions": 数组,长度{n-1},第k项评 第k张→第k+1张 的衔接:{{"from":k+1,"to":k+2,"level":"green|yellow|red","note":"理由+建议,一句话"}}。两张景别机位太接近=red/yellow并建议换;差异充分=green
- "overall": 整条片子的观感预判,2-3句
- "suggested_order": 给出最贴合歌词情绪推进的新顺序(如"1,3,2,4,...")和一句理由——把安静/叙事的歌词段配空镜或远景、情感高潮段配人物近景/特写,远近交替呼吸;已很好则写"当前顺序已合理"
- "motions": 数组,长度{n},第k项给"第k张图"的运镜建议:{{"img":k+1,"motion":"推近|拉远|左右移|上下移|放大横移","reason":"一句话"}}。原则:特写/近景人物→"推近"(向情绪聚焦);全景/建立镜头/宏大空镜→"拉远"或"左右移"(展开空间);中景→"推近"或轻"左右移";想强调纵深/天地→"上下移";开场大场面想兼顾推进与移动→"放大横移"。同类镜头的运镜也尽量错开,避免整片单调
只输出JSON,不要别的文字。"""
    raw = _ollama(prompt, timeout=900, cpu=True)   # 长推理走 CPU(护卡),给足超时
    result = _extract_json(raw) or {"raw": raw, "parse_error": True}
    return {"descriptions": descriptions, "result": result}


if __name__ == "__main__":
    import sys, time
    paths = sys.argv[1:]
    t = time.time()
    out = analyze_sequence(paths)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n耗时 {time.time()-t:.1f}s")
