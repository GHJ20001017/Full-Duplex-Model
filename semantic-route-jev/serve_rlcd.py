"""RLCD 0.6B 决策模型 —— 标准推理接口（Jev 契约兼容）。

对外契约与 Jev 的 POST /v1/systemone 保持一致：
    {state: {...}, questions: {id: {type, instructions, criteria}}}
    → {model, answers: {id: {type, choice/probabilities/confidence | noul}}}

支持的 question type:
  choice  —— 训练过的 3 类任务（continue/yield/wait），直接读 logits
  noul    —— **用同一个模型零样本做二选一**（A=true / B=false），
             不是训练目标，响应里明确标注 `trained: false`

## 分类体系与映射（重要）

模型训练的是三类 **"助手要不要继续当前播报"**：
    continue  用户让助手继续说 / 附和确认后不打断  → 继续播报
    yield     用户要求停止、纠正、接管话轮        → 让出话轮
    wait      用户话没说完 / 思考中                → 暂停等待

线上（demo / jev-duplex）问的是 **"助手要不要开口"**：reply / hold / wait。

这两套体系是同一决策的两种视角，对应关系是**语义映射、不是位置映射**：

    continue  →  hold    （继续播报 = 这一轮不另起回应 = 别开口）
    yield     →  reply   （让出话轮 = 现在轮到我说话 = 该回应）
    wait      →  wait

## 推理端如何知道该用哪套体系

**以 checkpoint 为准**，不再靠调用方猜：
    ckpt 目录里的 rlcd_config.json 记录 taxonomy / labels / system / user_tmpl，
    DecisionModel 启动时读它，提示词与槽位顺序都用 ckpt 自己的。
老 ckpt 没有该文件 → 回退 continue 体系并在响应里标注。

调用方即使给 speak 体系的 criteria（reply/hold/wait），也会按**语义名**
对到 ckpt 的槽位上（见 map_criteria_to_slots），不再是位置映射。
"""
import json
import os
import time

import numpy as np
import torch

import rlcd_taxonomies as _TAX

# 默认体系（模块级，供不加载模型时导入用）。真正的槽位顺序见 DecisionModel.slot_labels，
# 它来自 ckpt 的 rlcd_config.json —— 训练什么体系，推理就用什么，不靠调用方猜。
_DEFAULT_TAX = "continue"
LABELS = _TAX.get(_DEFAULT_TAX)["labels"]
LABEL_KEYS = LABELS

# 跨体系的**语义别名**：同一个决策在不同体系里的名字，不是位置关系。
# 用于把调用方 criteria 的键名对到本 ckpt 的槽位上。
#   continue(继续播报) ≈ hold(不出声)     两个体系在这里是"同类"
#   yield(让出话轮)    ≈ reply(该回应)
#   wait               ≈ wait
# canonical 语义组（与具体体系的名字无关）:
#   no_reply  这轮不另起回应   —— continue 体系叫 continue, speak 体系叫 hold
#   reply     该回应/让出话轮   —— continue 体系叫 yield,    speak 体系叫 reply
#   wait      继续听/别抢话     —— 两套都叫 wait
SEMANTIC_ALIAS = {
    "no_reply": ["continue", "hold", "no_reply", "silence", "don't_reply", "dont_reply"],
    "reply": ["yield", "reply", "respond", "answer"],
    "wait": ["wait", "listen", "hold_on", "incomplete"],
}

# 兼容旧引用：默认体系下的提示词
SYSTEM = _TAX.get(_DEFAULT_TAX)["system"]
USER_TMPL = _TAX.get(_DEFAULT_TAX)["user_tmpl"]

# noul 零样本模板：A=成立 / B=不成立，复用同样的 A/B 槽位
NOUL_SYSTEM = ("你只做判断，不解释。根据给定的对话内容判断一个陈述是否成立。"
               "只回答 A 或 B。")
NOUL_TMPL = """助手刚才在说：{assistant}

用户刚说：{user}

判断下面这个陈述是否成立：
"{claim}"

选项：
A. 成立
B. 不成立

答案："""


def map_criteria_to_slots(criteria_keys, slot_labels=None):
    """把调用方给的 criteria 键名，按**语义**对到本 ckpt 的槽位上。

    slot_labels: 本 ckpt 的标签顺序（默认取模块级 LABELS）。
    返回 (slot_index_per_key, unknown_keys)
    """
    slot_labels = list(slot_labels or LABELS)
    # 别名 -> 语义组
    alias2sem = {}
    for sem, names in SEMANTIC_ALIAS.items():
        for nm in names:
            alias2sem[nm] = sem
    # 语义组 -> 本 ckpt 的槽位下标：看本体系的哪个标签落在这个组里
    sem2slot = {}
    for i, lab in enumerate(slot_labels):
        g = alias2sem.get(str(lab).strip().lower())
        if g is not None:
            sem2slot[g] = i

    mapping, unknown = {}, []
    used = set()
    keys = list(criteria_keys)
    # 第一轮：按语义组命中
    for k in keys:
        kk = str(k).strip().lower()
        sem = alias2sem.get(kk)
        slot = sem2slot.get(sem) if sem is not None else None
        if slot is not None and slot not in used:
            mapping[k] = slot
            used.add(slot)
        else:
            mapping[k] = None
    # 第二轮：没命中的键，按顺序填补剩下的槽位（兜底，并记入 unknown）
    free = [i for i in range(len(slot_labels)) if i not in used]
    for k in keys:
        if mapping[k] is None:
            if free:
                mapping[k] = free.pop(0)
                unknown.append(k)
    return mapping, unknown


class DecisionModel:
    """RLCD 训练出的 0.6B 决策模型的最小推理封装。"""

    def __init__(self, ckpt, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
        self.model.to(self.device).eval()
        self.model.config.use_cache = False
        self.slot_ids = [self.tok.encode(k, add_special_tokens=False)[0] for k in ("A", "B", "C")]
        assert len(set(self.slot_ids)) == 3, "选项 token 冲突"

        # ---- 体系从 ckpt 读（训练什么，这里就用什么）----
        self.cfg = _TAX.load(ckpt)
        self.slot_labels = list(self.cfg["labels"])
        self.system = self.cfg["system"]
        self.user_tmpl = self.cfg["user_tmpl"]
        self.taxonomy = self.cfg["taxonomy"]
        if not self.cfg.get("_from_ckpt"):
            print("⚠️ ckpt 无 rlcd_config.json，按 continue 体系解释: %s" % ckpt)

    # ------------------------------------------------------------------ 内部
    def _encode(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        return self.tok.encode(p, add_special_tokens=False)

    @torch.no_grad()
    def _logits(self, seqs, n_slots):
        if not seqs:
            return np.zeros((0, n_slots))
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            logits = self.model(input_ids=ids, attention_mask=att).logits
        # 关键：批量时序列右侧 padding，[-1] 会落在 pad 上。
        # 必须按每条序列的真实长度取"最后一个有效位置"。
        last_idx = (att.sum(-1) - 1).clamp(min=0)
        rows = torch.arange(logits.size(0), device=logits.device)
        lg = logits[rows, last_idx][:, self.slot_ids[:n_slots]].float()
        return torch.softmax(lg, -1).cpu().numpy()

    # ------------------------------------------------------------------ 对外
    def predict(self, state, questions):
        """单条推理。state 可为 {user_said, assistant_said} 或 str。"""
        t0 = time.perf_counter()
        user, assistant = self._split_state(state)

        choice_qs = {k: v for k, v in questions.items() if v.get("type") == "choice"}
        noul_qs = {k: v for k, v in questions.items() if v.get("type") == "noul"}

        answers = {}

        # ---- choice：走训练过的路径 ----
        if choice_qs:
            seq = self._encode(self.system,
                               self.user_tmpl.format(assistant=assistant, user=user))
            P = self._logits([seq], 3)[0]
            for qid, q in choice_qs.items():
                keys = list((q.get("criteria") or {}).keys())
                if len(keys) == 3:
                    mp, unknown = map_criteria_to_slots(keys, self.slot_labels)
                    probs = {k: round(float(P[mp[k]]), 4) if mp[k] is not None else 0.0
                             for k in keys}
                    # argmax 必须在**调用方的键**上取，才不会错位
                    choice = max(probs, key=probs.get)
                    vals = np.array([probs[k] for k in keys], dtype=np.float64)
                    ent = -(vals * np.log(np.clip(vals, 1e-12, 1))).sum()
                    answers[qid] = {
                        "type": "choice", "choice": choice, "probabilities": probs,
                        "confidence": round(float(1 - ent / np.log(3)), 4),
                        "trained": True,
                        "taxonomy": self.taxonomy,
                    }
                    if unknown:
                        answers[qid]["trained"] = False
                        answers[qid]["note"] = ("criteria 键名 %s 无法按语义对上训练体系，"
                                               "已按顺序兜底" % unknown)
                else:
                    probs = {k: (round(float(P[i]), 4) if i < 3 else 0.0)
                             for i, k in enumerate(keys)}
                    answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get),
                                    "probabilities": probs,
                                    "confidence": round(float(max(probs.values(), default=0)), 4),
                                    "trained": False,
                                    "note": "checkpoint 只训练了 3 类，超出部分为占位"}

        # ---- noul：零样本二选一 ----
        for qid, q in noul_qs.items():
            claim = q.get("instructions", "")
            seq = self._encode(NOUL_SYSTEM,
                               NOUL_TMPL.format(assistant=assistant, user=user, claim=claim))
            p = self._logits([seq], 2)[0]
            answers[qid] = {"type": "noul", "noul": round(float(p[0]), 4),
                            "trained": False,
                            # ★ 明确告诉调用方：这个分数**不可用作阈值门**。
                            # 实测（display_path.py）：本模型的零样本 noul 与 Jev 几乎不相关
                            #   addressed    r=0.057   方向甚至是反的（该拦的 HOLD 打分更高）
                            #   is_complete  在"话没说完"样本上给 0.88（Jev 给 0.10）
                            # 用它做 compose() 的 <0.25 / <0.50 门，会把 raw 的 84% 砸到 44%。
                            "usable_for_gating": False,
                            "note": "零样本二选一（非训练目标），与 Jev 校准不一致，"
                                    "不可用于阈值门（详见 display_path.py）"}

        return {
            "model": "rlcd-qwen3-0.6b-%s" % self.taxonomy,
            "answers": answers,
            # 给调用方的用法提示：直接采信 action.choice；
            # noul 未训练，不要拿它们做 compose() 覆盖。
            "recommended_usage": {
                "use_action_directly": True,
                "gate_on_noul": False,
                "reason": "noul 头未训练，与 Jev 校准不一致；用其做阈值门会显著降低准确率",
            },
            "usage": {"output_tokens": 0},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    @staticmethod
    def _split_state(state):
        if isinstance(state, str):
            return state, ""
        user = state.get("user_said") or state.get("text") or ""
        assistant = state.get("assistant_said") or state.get("assistant") or ""
        return user, assistant

    def predict_batch_choice(self, pairs):
        """批量 action 推理 —— pairs: [(assistant, user), ...]。真正的吞吐杠杆。

        返回的列顺序恒为 LABELS（continue/yield/wait）。
        """
        seqs = [self._encode(self.system, self.user_tmpl.format(assistant=a, user=u))
                for a, u in pairs]
        return self._logits(seqs, 3)


if __name__ == "__main__":
    ckpt = os.environ.get("CKPT", "/tmp/laya_rlcd/ckpt_20e")
    m = DecisionModel(ckpt)
    print(f"模型已加载: {ckpt} -> {m.device}")
    for label, crit in [
        ("demo 的 speak 体系", {"reply": "该回应", "hold": "别开口", "wait": "继续听"}),
        ("训练体系原名", {"continue": "继续说", "yield": "让出话轮", "wait": "等待"}),
    ]:
        r = m.predict(
            {"assistant_said": "您这趟航班改签到明天下午需要补八十元差价，另外还有二十五元改签服务费。",
             "user_said": "不用停你接着说改签费怎么算"},
            {"action": {"type": "choice", "instructions": "助手接下来应该做什么？",
                        "criteria": crit}})
        print(f"  [{label}] {json.dumps(r['answers']['action'], ensure_ascii=False)}")
