#!/usr/bin/env python3
"""vision_advisor.py — 用本地 Ollama gemma4 多模态对多机位排序做语义级预判。

两段式:
  1) describe_image: 逐图理解(景别/机位角度/人物姿态/构图/光线),结构化一句话。
  2) analyze_sequence: 把按序描述 + 歌曲情绪 + 剪辑原则喂给 gemma4(纯文本),
     产出首图/尾图评估、逐段衔接红黄绿+建议、整体观感预判、建议排序。

被 app.py 调用(也可 `python3 vision_advisor.py img1 img2 ...` 单测)。
gemma4 是 thinking 模型,务必 think=False(否则输出夹带推理)。
"""
import base64, json, os, re, urllib.request

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma4:12b-it-qat")


def _ollama(prompt, images=None, timeout=180):
    payload = {"model": VISION_MODEL, "prompt": prompt, "stream": False, "think": False}
    if images:
        payload["images"] = images
    req = urllib.request.Request(f"{OLLAMA}/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
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


def analyze_sequence(image_paths, song_hint=""):
    """返回 dict:{descriptions:[...], result:{first,last,transitions[],overall,suggested_order}}"""
    descriptions = [describe_image(p) for p in image_paths]
    n = len(descriptions)
    listing = "\n".join(f"第{i+1}张: {d}" for i, d in enumerate(descriptions))
    hint = song_hint or "一首怀旧温暖的男女对唱(圆舞曲),情绪曲线:前奏安静→主歌收敛→副歌放开→结尾归于平静。"
    prompt = f"""你是资深MV剪辑指导。下面是一首歌按当前顺序排好的 {n} 个机位画面的描述。
歌曲背景:{hint}

机位描述(当前顺序):
{listing}

请按"剪辑30度规则(相邻镜头要换景别或转30度以上机位,否则像跳切)"和"情绪曲线"评估这个排序,只输出一个JSON对象,字段:
- "first": 首图是否适合开场(要有建立镜头/较大景别/安定感),一句话评价+建议
- "last": 尾图是否适合收尾(要有收束/归于平静感),一句话评价+建议
- "transitions": 数组,长度{n-1},第k项评 第k张→第k+1张 的衔接:{{"from":k+1,"to":k+2,"level":"green|yellow|red","note":"理由+建议,一句话"}}。两张景别机位太接近=red/yellow并建议换;差异充分=green
- "overall": 整条片子的观感预判,2-3句
- "suggested_order": 若当前顺序可优化,给出建议的新顺序(如"1,3,2,4,...")和一句理由;已很好则写"当前顺序已合理"
只输出JSON,不要别的文字。"""
    raw = _ollama(prompt, timeout=240)
    result = _extract_json(raw) or {"raw": raw, "parse_error": True}
    return {"descriptions": descriptions, "result": result}


if __name__ == "__main__":
    import sys, time
    paths = sys.argv[1:]
    t = time.time()
    out = analyze_sequence(paths)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n耗时 {time.time()-t:.1f}s")
