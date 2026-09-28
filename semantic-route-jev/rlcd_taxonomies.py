"""两套分类体系的**单一事实来源**。

背景（为什么需要这个文件）：
本项目的 0.6B 模型先后被用在两个**不同**的决策上，一开始混用了同一个提示词，
导致训出来的模型和线上要的行为对不上（见 speak_agreement.py：366 条里只有 64.2% 一致）。

  continue 体系 —— "助手要不要继续当前播报"（最初训练用）
      continue  用户让助手继续说 / 附和确认  → 继续播报
      yield     用户要求停止、纠正、接管话轮  → 让出话轮
      wait      用户话没说完 / 思考中         → 暂停等待

  speak 体系 —— "助手要不要开口"（demo / jev-duplex 线上契约）
      reply     用户在对助手提问/指令/求确认/纠错 → 立刻回应
      hold      附和、确认、寒暄、大笑、自言自语、对第三人说 → 不出声
      wait      话明显没说完                     → 继续听

两者的实质差别不是命名，而是 **continue 把 speak 的两类合并了**：
  continue ≈ {reply: "接着说", hold: "附和确认"} —— 104 / 23 条（见 speak_agreement.json）
所以不能靠映射互换，必须按目标体系训练。

任何脚本（train / eval / cv / serve）都从这里取提示词与标签顺序，
checkpoint 保存时把 taxonomy 名字一并写入 rlcd_config.json，
推理端读 ckpt 的配置 —— 杜绝"训练一个体系、推理另一个体系"再次发生。
"""
import json
import os

TAXONOMIES = {
    "continue": {
        "name": "continue",
        "title": "助手是否继续当前播报",
        "labels": ["continue", "yield", "wait"],
        "labels_upper": ["CONTINUE", "YIELD", "WAIT"],
        "system": (
            "你在判断全双工语音对话中的下一步动作。助手正在播报，用户插话说了一句话（ASR转写，"
            "可能无标点、有口误）。判断助手接下来应该做什么。只回答一个编号。"
        ),
        "user_tmpl": """助手刚才在说：{assistant}

用户刚说：{user}

选项：
A. 继续 —— 用户表示请继续说、附和确认，或在不打断播报的前提下追问
B. 让出 —— 用户明确要求停止、纠正助手刚说的内容、打断并接管话轮
C. 等待 —— 用户话没说完、思考中、或自我修正后意图仍不明确

答案：""",
    },
    "speak": {
        "name": "speak",
        "title": "助手是否该开口回应",
        "labels": ["reply", "hold", "wait"],
        "labels_upper": ["REPLY", "HOLD", "WAIT"],
        "system": (
            "你在判断全双工语音对话中，语音助手接下来应该做什么。助手正在说话，"
            "用户刚说了一句（ASR转写，可能无标点、有口误）。判断助手现在是否该开口。只回答一个编号。"
        ),
        "user_tmpl": """助手刚才在说：{assistant}

用户刚说：{user}

选项：
A. 回应 —— 用户在向助手提问、下指令、求确认、纠错或抱怨需要处理，应该立刻回应
B. 不出声 —— 用户只是在附和、确认、寒暄、大笑、自言自语，或在跟旁边的人说话
C. 等待 —— 用户话明显还没说完，助手应继续听，不要抢话

答案：""",
    },
}

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
