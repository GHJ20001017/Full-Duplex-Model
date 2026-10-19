"""Training-local taxonomy snapshot used by V7-B. Checkpoints save the exact prompt."""
import json
import os

TAXONOMIES = {'continue': {'name': 'continue', 'title': '助手是否继续当前播报', 'labels': ['continue', 'yield', 'wait'], 'labels_upper': ['CONTINUE', 'YIELD', 'WAIT'], 'system': '你在判断全双工语音对话中的下一步动作。助手可能正在播报，也可能尚未开口；助手文本为空表示尚无助手输出。ASR片段可能是用户发言，也可能是助手语音回录，可能有同音误写和口误。结合上下文判断，只回答一个编号。', 'user_tmpl': '助手文本（空表示尚无输出）：{assistant}\n\nASR片段：{user}\n\n选项：\nA. 继续 —— 用户请继续、附和，或不打断地追加提问；片段若是助手语音回录，不因回录打断\nB. 让出 —— 用户明确停止、纠错、接管话轮或提出需要回应的请求；助手尚未开口时接收明确请求，不表示必须取消正在生成的回复\nC. 等待 —— 用户话未说完、思考中或自我修正后意图不明确\n\n答案：'}, 'speak': {'name': 'speak', 'title': '助手是否该开口回应', 'labels': ['reply', 'hold', 'wait'], 'labels_upper': ['REPLY', 'HOLD', 'WAIT'], 'system': '你在判断全双工语音对话中，语音助手接下来应该做什么。助手正在说话，用户刚说了一句（ASR转写，可能无标点、有口误）。判断助手现在是否该开口。只回答一个编号。', 'user_tmpl': '助手刚才在说：{assistant}\n\n用户刚说：{user}\n\n选项：\nA. 回应 —— 用户在向助手提问、下指令、求确认、纠错或抱怨需要处理，应该立刻回应\nB. 不出声 —— 用户只是在附和、确认、寒暄、大笑、自言自语，或在跟旁边的人说话\nC. 等待 —— 用户话明显还没说完，助手应继续听，不要抢话\n\n答案：'}}

DEFAULT = os.environ.get("TAXONOMY", "continue")


def get(name=None):
    """按名字取体系配置；名字为空时用环境变量 TAXONOMY（默认 continue）。"""
    n = (name or DEFAULT or "continue").strip().lower()
    if n not in TAXONOMIES:
        raise KeyError("未知 taxonomy=%r，可选: %s" % (n, list(TAXONOMIES)))
    return TAXONOMIES[n]


CONFIG_FILENAME = "rlcd_config.json"


def save(ckpt_dir, taxonomy_name, extra=None):
    """把体系写进 checkpoint 目录，供推理端读取。"""
    t = get(taxonomy_name)
    cfg = {
        "taxonomy": t["name"],
        "title": t["title"],
        "labels": t["labels"],
        "labels_upper": t["labels_upper"],
        "system": t["system"],
        "user_tmpl": t["user_tmpl"],
        "slots": ["A", "B", "C"],
    }
    if extra:
        cfg.update(extra)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, CONFIG_FILENAME)
    json.dump(cfg, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return path


def load(ckpt_dir):
    """读 checkpoint 的体系配置；老 ckpt 没有该文件时回退到 continue（并标注）。"""
    path = os.path.join(ckpt_dir, CONFIG_FILENAME)
    if os.path.exists(path):
        cfg = json.load(open(path, encoding="utf-8"))
        cfg["_from_ckpt"] = True
        return cfg
    t = get("continue")
    return {"taxonomy": "continue", "title": t["title"], "labels": t["labels"],
            "labels_upper": t["labels_upper"], "system": t["system"],
            "user_tmpl": t["user_tmpl"], "slots": ["A", "B", "C"],
            "_from_ckpt": False,
            "_note": "老 checkpoint 无 rlcd_config.json，按 continue 体系解释"}
