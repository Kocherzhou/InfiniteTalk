#!/usr/bin/env python3
"""vision_advisor.py — 用本地 Ollama gemma4 多模态对多机位排序做语义级预判。

两段式:
  1) describe_image: 逐图理解(景别/机位角度/人物姿态/构图/光线),结构化一句话。
  2) analyze_sequence: 把按序描述 + 歌曲情绪 + 剪辑原则喂给 gemma4(纯文本),
     产出首图/尾图评估、逐段衔接红黄绿+建议、整体观感预判、建议排序。

被 app.py 调用(也可 `python3 vision_advisor.py img1 img2 ...` 单测)。
gemma4 是 thinking 模型,务必 think=False(否则输出夹带推理)。
"""
import base64, hashlib, json, os, re, subprocess, time, urllib.request

_DESC_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desc_cache")

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
    data = open(path, "rb").read()
    cf = os.path.join(_DESC_CACHE, hashlib.md5(data).hexdigest() + ".txt")
    if os.path.exists(cf):                       # 按图内容缓存:同图重跑跳过 GPU 描述
        try:
            return open(cf, encoding="utf-8").read()
        except Exception:
            pass
    p = ("用中文简洁描述这张MV机位图,只客观描述不发挥,一行内涵盖:"
         "景别(特写/近景/中景/全景)、机位角度(正面/侧面/俯/仰/平视)、"
         "人物数量与姿态朝向、构图、光线明暗。")
    desc = _ollama(p, images=[base64.b64encode(data).decode()])
    try:
        os.makedirs(_DESC_CACHE, exist_ok=True)
        open(cf, "w", encoding="utf-8").write(desc)
    except Exception:
        pass
    return desc


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


def plan_shots(image_paths, slots, song_hint="", progress_cb=None):
    """分镜排期师:给【结构镜头槽】(每槽含段落名/时长/空镜标记)分配 图+运镜。
    slots: [{i,start,end,dur,section,role}]  role='空镜'强制用无人画面。
    返回 {descriptions, plan:[{slot,img,motion,reason}]}。
    逐图理解走 GPU+节流(短脉冲),排期长推理走 CPU(护卡)。"""
    n = len(image_paths)
    descriptions = []
    for i, p in enumerate(image_paths):
        if progress_cb:
            progress_cb("describe", i, n)
        descriptions.append(describe_image(p))
    if progress_cb:
        progress_cb("reason", n, n)
    imgs = "\n".join(f"图{i+1}: {d}" for i, d in enumerate(descriptions))
    slotlist = "\n".join(
        f"槽{s['i']}: {s['start']:.0f}-{s['end']:.0f}s({s['dur']:.0f}s) 段落[{s['section']}]"
        + ("　【必须空镜/无人画面】" if s.get("role") == "空镜" else "")
        for s in slots)
    m = len(slots)
    prompt = f"""你是资深MV分镜导演。一首歌已按【音乐结构】切成 {m} 个镜头槽(每槽是一个乐句段落、长度不一),
歌曲信息/歌词:
{song_hint}

镜头槽(按时间顺序,共{m}个):
{slotlist}

可用画面素材({n}张,可重复使用):
{imgs}

请为【每一个槽】分配一张图和一个运镜,排出完整分镜表。规则:
- 标【必须空镜】的槽(前奏/尾奏)只能用"无人物/纯风景空镜"的图;其余槽优先用对应情绪的画面。
- 图可重复(把关键画面当"视觉动机"),但**相邻两槽不要用同一张**(除非刻意延续)。**复用优先级:人物"弹唱位/演唱位"这类标准表演镜头是MV的视觉锚点,反复出现不疲劳,优先拿它复用(尤其副歌/华彩反复回到弹唱位);其次其他人物镜头;空镜/风景则尽量每个都不同、保持新鲜,别重复用同一张空镜(前奏与尾奏首尾呼应可例外)。**
- 副歌/华彩(Chorus/Bridge/Final)配最有情感张力的人物近景/特写;主歌(Verse)配叙事中景;段落内被拆成 (1/2)(2/2) 的,两槽尽量换不同图或不同运镜,制造推进。
- 运镜五选一:推近(情绪聚焦)/拉远(展开空间·宏大)/左右移(横向展开)/上下移(纵深天地)/放大横移(开场大场面)。同段相邻槽运镜尽量错开。
只输出一个JSON对象:{{"plan":[{{"slot":0,"img":3,"motion":"拉远","reason":"一句话"}}, ...]}},plan 数组长度必须正好 {m}、slot 从0到{m-1}按序、img 是1到{n}的整数。不要别的文字。"""
    def _good_plan(raw):
        obj = _extract_json(raw)
        plan = obj.get("plan") if isinstance(obj, dict) else (obj if isinstance(obj, list) else None)
        if isinstance(plan, list) and len(plan) >= 1 and all(isinstance(x, dict) and "img" in x for x in plan):
            # 补全/钳制:slot 缺则按序、img 钳到 1..n、motion 落到合法集
            for k, p in enumerate(plan):
                p["slot"] = p.get("slot", k)
                try: p["img"] = min(n, max(1, int(p.get("img", 1))))
                except Exception: p["img"] = 1
                if p.get("motion") not in ("推近", "拉远", "左右移", "上下移", "放大横移"):
                    p["motion"] = "推近"
            return plan
        return None
    plan = None
    for _ in range(3):                       # gemma 偶发坏 JSON → 最多重试 3 次(CPU)
        plan = _good_plan(_ollama(prompt, timeout=900, cpu=True))
        if plan:
            break
    if not plan:
        raise RuntimeError("军师分镜输出解析失败(gemma JSON 3 次都不合法),请重试")
    return {"descriptions": descriptions, "plan": plan}


if __name__ == "__main__":
    import sys, time
    paths = sys.argv[1:]
    t = time.time()
    out = analyze_sequence(paths)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n耗时 {time.time()-t:.1f}s")
