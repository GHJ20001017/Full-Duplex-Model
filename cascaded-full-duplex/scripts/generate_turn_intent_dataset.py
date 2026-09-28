"""Generate four-field turn-intent JSONL through an OpenAI-compatible Chat Completions API.

Examples (run from the project root):
  python scripts/generate_turn_intent_dataset.py --count 300 --dry-run
  python scripts/generate_turn_intent_dataset.py --model MODEL --count 3000 \
      --base-url https://YOUR-ENDPOINT/v1 --output turn_intent.jsonl

Set OPENAI_API_KEY (or select an environment variable with --api-key-env).
Output parents must exist. Existing files are never overwritten; no implicit resume.
A companion .manifest.jsonl records generation groups for grouped dataset splitting.
Synthetic output is NOT human-reviewed; structural checks do not prove label correctness.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

LABELS = ("YIELD", "CONTINUE", "WAIT")
FIELDS = {"assistant_text", "user_asr", "label", "reason"}
# These are targeted acoustic/turn-taking cases that ordinary topic contrasts miss.
SPECIAL_CASES = ("normal", "wake_start", "assistant_echo")
DOMAINS = ("日常闲聊", "天气出行", "餐饮购物", "旅游规划", "技术咨询", "课程讲解", "健康常识", "办公安排")
TOPIC_FAMILIES = {
    "出行交通": ("航班改签", "火车票退票", "公交换乘", "酒店入住"),
    "消费服务": ("餐厅预订", "网购退货", "快递配送", "售后维修"),
    "工作协作": ("会议时间调整", "项目进度确认", "任务分工安排", "邮件内容确认"),
    "学习解释": ("代码报错排查", "数学概念讲解", "英语语法说明", "实验步骤说明"),
    "生活提醒": ("天气变化提醒", "药品服用说明", "家电使用说明", "做饭步骤提醒"),
    "信息查询": ("营业时间查询", "路线信息确认", "价格费用说明", "地址信息确认"),
}
TOPICS = tuple(topic for topics in TOPIC_FAMILIES.values() for topic in topics)
TOPIC_TO_FAMILY = {topic: family for family, topics in TOPIC_FAMILIES.items() for topic in topics}
STYLES = ("简短直接", "委婉口语", "自然长句", "自我修正", "带少量中英混合术语")
ASR_STYLES = ("自然转写", "无标点转写", "口头重复", "少量合理同音误识别")
ASSISTANT_LENGTH_HINTS = ("短一些", "中等长度", "较长一些")
CONTEXTS = ("助手输出短文本", "助手输出中等长度上下文", "助手输出较长上下文")
ASSISTANT_SURFACES = ("完整句带句末标点", "无句末标点的文本片段", "自然截断的未完句")
ASR_PUNCTUATION = ("无标点", "中文逗号", "中文句号", "问号", "感叹号", "逗号与问号混用")
BEHAVIORS = {
    "YIELD": ("明确停止", "无停止关键词的纠正", "改变回答方向", "追问", "附和后转折提出要求", "回答助手确认问题"),
    "CONTINUE": ("简短附和", "长句认同并要求继续", "否定停止如不用停", "允许继续讲解", "确认听懂但不接管话轮"),
    "WAIT": ("信息尚未足够但没有明确接管", "缺少指代对象的歧义", "用户改口尚未完成", "只有话题起始词", "局部上下文不足"),
}
ELLIPSIS_RE = re.compile(r"(?:\.{2,}|[…⋯⋮︙]+)")

SYSTEM = """你在构建实时语音助手的语义打断训练数据，不是在回答用户问题。
适用范围：包括助手尚未开始输出、助手正在播报且用户开始插话。分类器只能看到 assistant_text 和 user_asr。
标签定义：
YIELD：当前可见信息已足以要求助手停止当前播报、让出话轮；在助手尚未开始输出的唤醒起始阶段，只要用户已有可识别发言，也标为YIELD。
CONTINUE：当前明确是附和、认同或要求助手继续；如果user_asr只是重复assistant_text开头的几个字，视为助手音频回声/自回录，也标为CONTINUE，而不是用户接管。
WAIT：仅凭当前输入无法确定意图；不是所有未完成句子都必须WAIT。
输入口径：
assistant_text是该次判断时刻服务端已输出的助手文本，不保证全部已被用户听到，不含未来生成内容；唤醒起始阶段可以为空字符串。
user_asr是该次用户插话的当前累计ASR假设，不是未来完整句子，也不是只有最后新增的字。
没有历史对话、音频、语气、is_final或隐藏意图可用。标签和原因只能依据这两个文本字段。
关键特殊场景：
1. wake_start：assistant_text必须为空，user_asr是用户在助手首字输出前已经说出的可识别内容，label必须是YIELD。不要因为assistant_text为空把它标成WAIT。
2. assistant_echo：assistant_text必须是自然助手话语，user_asr是同一段助手音频被回录后的“连续时间前缀”，不是用户主动复述。它应从assistant_text开头连续截取一小段（不要跳过中间词、拼接远处词），通常3到12个汉字；前面一两个字应与assistant_text相同，后面允许出现一个同音字、近音字或单字识别错误。不要生成完全正确的复制，也不要生成“明天天气”这种跳过“北京”的非连续片段。只模拟回声，不加入新的用户请求，label必须是CONTINUE。不要把这个短前缀当成用户确认或接管。
数据规则：
1. 不靠停、对、嗯等关键词机械分类。覆盖否定、转折、纠正、追问和上下文相关表达。
2. 助手讲解时的“对”可能是附和；回答助手确认问题的“对”可能需要让出话轮。
3. “嗯，我想”证据不足；“嗯，我想换个问题”已有明确意图。不能用未来词给前缀倒标。
4. 模拟自然口语和合理ASR错误，不随机破坏字符。按实际可见转写标注，不按不可见原音标注。
5. 不要让领域、长度、口语风格与某个标签固定绑定。不要仅替换人名地名复制同一模板。
6. WAIT必须有具体信息不足的原因，不能用于掩盖标签冲突；明确要求停止的半句话也可YIELD。
7. reason用一句简短中文指出输入中的关键证据，不编造情绪/语气/历史，不重复标签或写长篇推理。
8. 各字段不能包含<think>或<answer>等输出格式标记。reason与label分开存储，训练时再组合。
9. 所有文本字段禁止省略号，包括中文省略号、连续英文句点以及省略号的Unicode变体。未完成的ASR直接结束在当前词，不添加截断标记；不要用重复填充词凑长度。
10. 同一topic_group使用相同的具体话题和基本场景，针对不同label构造纠正、否定、附和、指代不明等对照。可调整助手是在讲解还是提问，不能靠话题本身决定标签。
11. user_asr和assistant_text长度按槽位随机变化；assistant_context只提供短、中等、较长的软偏好，不要求精确字符数，也不能因为差几个字拒收。仍需遵守assistant_text和user_asr的最大字符上限。不要让长度与label固定绑定。
12. assistant_surface指定助手文本形态。自然截断的未完句必须直接停在词或句子中间，例如“改签费是八十元，支”，不补全后续，不加句末标点或省略号；标签只依据当前可见片段。不要先用完整回答标注后再机械截断。无句末标点的文本片段也允许完全没有标点。
13. user_punctuation指定当前ASR标点风格：无标点/中文逗号/中文句号/问号/感叹号/逗号与问号混用。按槽位自然表达，不为凑标点编造请求；句号不表示用户说完，问号不自动意味着YIELD。三类标签都可有或没有句末标点。asr_style只控制词汇转写，标点以user_punctuation为准。
14. topic_family用于数据切分，scenario_seed表示同一具体场景；相同scenario_seed的三个槽位必须保持同一事件和助手上下文，只改变用户交互行为与自然表达。不要让topic_family或topic单独决定label。
15. 每个topic_family和具体topic都必须覆盖YIELD、CONTINUE、WAIT；每种label也必须出现在多个topic_family。behavior是目标交互行为提示，不得用固定关键词替代语义判断。
16. assistant_surface、user_punctuation与label交叉组合；不要让截断、无标点、问号、句号、短输入或长输入固定对应某个label。
17. 长上下文必须有连贯且相关的信息，不重复填充，不机械截断来凑长度；长度偏好只是随机提示，不是验收条件。
遵循请求中的槽位编号、标签和场景约束，但必须构造语义确实符合标签的输入，不要牵强解释。
只返回一个JSON对象：{"samples":[{"slot_id":0,"assistant_text":"助手文本","user_asr":"当前转写","label":"YIELD","reason":"可见判断依据"}]}。
不要Markdown围栏、额外字段或正文。slot_id只用于匹配生成计划，最终数据将删除它。
"""


def load_existing_data(path):
    """Load prior JSONL rows so resume mode can continue line numbers."""
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != FIELDS:
                    raise ValueError(f"Invalid existing data at line {line_number}")
                if any(not isinstance(row[key], str) for key in FIELDS):
                    raise ValueError(f"Invalid existing data types at line {line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read existing data: {path} ({type(exc).__name__})") from exc
    return rows


def make_plan(count, seed, topic=None, max_asr_chars=300, max_assistant_chars=1200,
              special_cases=()):
    if max_assistant_chars <= 0:
        raise ValueError("max_assistant_chars must be positive")
    rng = random.Random(seed)
    rng = random.Random(seed)
    available_topics = TOPICS
    special_cases = tuple(dict.fromkeys(special_cases))
    invalid_special_cases = set(special_cases) - set(SPECIAL_CASES[1:])
    if invalid_special_cases:
        raise ValueError(f"Unknown special case: {sorted(invalid_special_cases)}")
    plan = []
    group_count = count if special_cases else (count + 2) // 3
    for group in range(group_count):
        if special_cases:
            labels = ["YIELD" if special_cases[group % len(special_cases)] == "wake_start" else "CONTINUE"]
        else:
            labels = list(LABELS)
            rng.shuffle(labels)
        if topic is None:
            group_topic = available_topics[group % len(available_topics)]
            family = TOPIC_TO_FAMILY[group_topic]
        else:
            group_topic = topic
            family = TOPIC_TO_FAMILY.get(topic, "自定义主题")
        scenario_seed = f"seed{seed}_scene{group}_{family}"
        length_hint = rng.choice(ASSISTANT_LENGTH_HINTS)
        assistant_context = rng.choice(CONTEXTS)
        if special_cases:
            special_case = special_cases[group % len(special_cases)]
        else:
            special_case = "normal"
        for label in labels:
            if len(plan) == count:
                break
            slot_special_case = special_case
            if special_case == "wake_start" and label != "YIELD":
                slot_special_case = "normal"
            elif special_case == "assistant_echo" and label != "CONTINUE":
                slot_special_case = "normal"
            plan.append({
                "slot_id": len(plan),
                "topic_group": f"seed{seed}_topic{group}",
                "topic_family": family,
                "topic": group_topic,
                "scenario_seed": scenario_seed,
                "label": label,
                "behavior": rng.choice(BEHAVIORS[label]),
                "special_case": slot_special_case,
                "style": rng.choice(STYLES),
                "asr_style": rng.choice(ASR_STYLES),
                "assistant_surface": ASSISTANT_SURFACES[(group + len(plan)) % len(ASSISTANT_SURFACES)],
                "user_punctuation": ASR_PUNCTUATION[len(plan) % len(ASR_PUNCTUATION)],
                "assistant_context": assistant_context,
                "length_hint": length_hint,
            })
    return plan


def select_batch(pending, batch_size):
    """Pack whole remaining topic groups without exceeding the request limit."""
    slots = []
    for start in range(0, len(pending)):
        slot = pending[start]
        if start and slot["topic_group"] == pending[start - 1]["topic_group"]:
            continue
        group = []
        for candidate in pending[start:start + 3]:
            if candidate["topic_group"] != slot["topic_group"]:
                break
            group.append(candidate)
        if len(slots) + len(group) > batch_size:
            break
        slots.extend(group)
    return slots


def make_messages(slots, recent, max_assistant_chars, max_asr_chars, siblings=None):
    request = {
        "slots": slots,
        "length_limits": {"assistant_text": max_assistant_chars, "user_asr": max_asr_chars, "reason": 120},
        "instructions": "每个槽位生成一个样本。相同scenario_seed的槽位必须保持同一具体事件和助手上下文，分别实现目标label的对照；topic_family和topic不能决定label。special_case是必须遵守的场景约束：wake_start只用于YIELD，assistant_text必须为空且user_asr必须是用户已说出的内容；assistant_echo只用于CONTINUE，assistant_text必须非空，user_asr必须是assistant_text开头连续截取的3到12个汉字，前1到2个字保持一致，后续只允许一个轻微同音/近音/单字ASR错误，不能完全正确复制，不能跳过中间词拼接远处文字，也不能加入新的用户意图。normal按普通语义打标。长度偏好和assistant_context只是软提示，请自然随机决定assistant_text与user_asr的实际长度，不要为了达到某个字符数填充或拒绝；只需遵守最大字符上限。按assistant_surface生成完整、有标点或自然截断的assistant_text；按user_punctuation自然变化user_asr，标点不是标签信号。不要复用近期输入，不要生成省略号。",
        "accepted_topic_siblings": siblings or [],
        "retry_instruction": "若有同组已接受样本，保持其具体场景，只补缺失槽位；允许复用助手上下文，但不要复制完整输入对。",
        "recent_inputs_to_avoid": recent[-12:],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]


def fingerprint(row):
    def normalize(text):
        text = unicodedata.normalize("NFKC", text).casefold()
        return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))
    return normalize(row["assistant_text"]), normalize(row["user_asr"])


def is_contiguous_echo_with_minor_asr_error(assistant_text, user_asr):
    """Accept a short contiguous replay prefix with a small recognition error."""
    assistant = "".join(c for c in unicodedata.normalize("NFKC", assistant_text) if not unicodedata.category(c).startswith("P"))
    user = "".join(c for c in unicodedata.normalize("NFKC", user_asr) if not unicodedata.category(c).startswith("P"))
    if not 3 <= len(user) <= 12 or len(user) >= len(assistant):
        return False
    # Echo must stay aligned to the assistant's beginning: no skipped words.
    if assistant[:2] != user[:2]:
        return False
    mismatches = sum(a != b for a, b in zip(assistant, user))
    return mismatches == 1


def validate_samples(content, slots, seen, max_assistant_chars=1200, max_asr_chars=300):
    """Validate structure and exact normalized input duplicates, NOT semantic truth."""
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {"samples"} or not isinstance(payload["samples"], list):
        raise ValueError("Expected an object containing only a samples array")
    expected = {slot["slot_id"]: slot for slot in slots}
    accepted = []
    rejected = Counter()
    ids = set()
    local_seen = set(seen)
    for row in payload["samples"]:
        if not isinstance(row, dict) or set(row) != FIELDS | {"slot_id"}:
            rejected["fields"] += 1
            continue
        ident = row["slot_id"]
        if type(ident) is not int or ident not in expected or ident in ids:
            rejected["slot_id"] += 1
            continue
        if any(not isinstance(row[key], str) for key in FIELDS):
            rejected["types"] += 1
            continue
        clean = {key: row[key].strip() for key in ("assistant_text", "user_asr", "label", "reason")}
        if clean["label"] != expected[ident]["label"]:
            rejected["label"] += 1
            continue
        if (not clean["assistant_text"] and expected[ident].get("special_case") != "wake_start"):
            rejected["empty_assistant"] += 1
            continue
        if expected[ident].get("special_case") == "wake_start":
            if clean["label"] != "YIELD" or not clean["user_asr"] or clean["assistant_text"]:
                rejected["wake_start"] += 1
                continue
        elif expected[ident].get("special_case") == "assistant_echo":
            if (clean["label"] != "CONTINUE"
                    or not is_contiguous_echo_with_minor_asr_error(
                        clean["assistant_text"], clean["user_asr"])):
                rejected["assistant_echo"] += 1
                continue
        if ((not clean["assistant_text"] and expected[ident].get("special_case") != "wake_start")
                or len(clean["assistant_text"]) > max_assistant_chars
                or len(clean["user_asr"]) > max_asr_chars or not 4 <= len(clean["reason"]) <= 120):
            rejected["length"] += 1
            continue
        if any(ELLIPSIS_RE.search(unicodedata.normalize("NFKC", value)) for value in clean.values()):
            rejected["ellipsis"] += 1
            continue
        if not clean["user_asr"] and clean["label"] != "WAIT":
            rejected["empty_asr"] += 1
            continue
        if any(re.search(r"</?(?:answer|think)\b", value, re.I) for value in clean.values()):
            rejected["format_tags"] += 1
            continue
        key = fingerprint(clean)
        if key in local_seen:
            rejected["duplicate_input"] += 1
            continue
        ids.add(ident)
        local_seen.add(key)
        accepted.append((ident, clean))
    return accepted, rejected


def generate(client, args, data_file, manifest_file, initial_rows=()):
    pending = make_plan(args.count, args.seed, args.topic, args.max_asr_chars, args.max_assistant_chars,
                        args.special_cases)
    seen = {fingerprint(row) for row in initial_rows}
    recent = [{"topic_group": "previous",
               "assistant_text": row["assistant_text"], "user_asr": row["user_asr"]}
              for row in initial_rows[-12:]]
    group_examples = {}
    total = len(initial_rows)
    label_counts = Counter(row["label"] for row in initial_rows)
    for attempt in range(1, args.max_requests + 1):
        if not pending:
            break
        slots = select_batch(pending, args.batch_size)
        siblings = [example for group in dict.fromkeys(s["topic_group"] for s in slots)
                    for example in group_examples.get(group, [])]
        try:
            kwargs = {
                "model": args.model,
                "messages": make_messages(slots, recent, args.max_assistant_chars, args.max_asr_chars, siblings),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise ValueError("Incomplete or non-text completion")
            content = choice.message.content
            if not isinstance(content, str):
                raise ValueError("Missing text content")
            accepted, rejected = validate_samples(
                content, slots, seen, args.max_assistant_chars, args.max_asr_chars
            )
        except (ValueError, IndexError, TypeError):
            print(f"request={attempt}: invalid/incomplete JSON response; retrying remaining slots", file=sys.stderr)
            continue
        except Exception as exc:
            # Do not log provider exception text: it can contain request data or credentials.
            status = getattr(exc, "status_code", None)
            retryable = status in (408, 409, 429) or (isinstance(status, int) and status >= 500)
            retryable = retryable or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
            if not retryable:
                print(f"API failure: {type(exc).__name__}, status={status}; partial output retained", file=sys.stderr)
                return 1
            print(f"request={attempt}: transient API failure ({type(exc).__name__}, status={status})", file=sys.stderr)
            if attempt < args.max_requests:
                time.sleep(min(2 ** min(attempt, 5), 30))
            continue
        accepted_ids = set()
        slot_map = {slot["slot_id"]: slot for slot in slots}
        for ident, row in accepted:
            data_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_file.write(json.dumps({
                "line": total + 1,
                "generation_group": slot_map[ident]["topic_group"],
                "request_attempt": attempt,
                "slot": slot_map[ident],
                "model": args.model,
                "source": "synthetic_unreviewed",
            }, ensure_ascii=False) + "\n")
            seen.add(fingerprint(row))
            recent.append({"topic_group": slot_map[ident]["topic_group"],
                           "assistant_text": row["assistant_text"], "user_asr": row["user_asr"]})
            recent = recent[-12:]
            group_examples.setdefault(slot_map[ident]["topic_group"], []).append(
                {"topic_group": slot_map[ident]["topic_group"], **row})
            accepted_ids.add(ident)
            label_counts[row["label"]] += 1
            total += 1
        data_file.flush()
        manifest_file.flush()
        pending = [slot for slot in pending if slot["slot_id"] not in accepted_ids]
        unfinished_groups = {slot["topic_group"] for slot in pending}
        group_examples = {group: examples for group, examples in group_examples.items() if group in unfinished_groups}
        print(f"request={attempt}: accepted={len(accepted)} total={total}/{args.count} rejected={dict(rejected)}", file=sys.stderr)
    print(json.dumps({"written": total, "requested": args.count, "labels": dict(label_counts),
                      "complete": not pending}, ensure_ascii=False))
    return 0 if not pending else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Provider model ID; required except in dry-run")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, default=Path("turn_intent.jsonl"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--topic", help="Optional shared topic, e.g. 航班改签; otherwise rotate topic families")
    parser.add_argument("--special-case", dest="special_cases", action="append", choices=SPECIAL_CASES[1:], default=[],
                        help="Generate only targeted cases; repeat to alternate wake_start and assistant_echo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, help="Hard cap on API calls including invalid outputs and retries")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-assistant-chars", type=int, default=1200)
    parser.add_argument("--max-asr-chars", type=int, default=300)
    parser.add_argument("--json-mode", action="store_true", help="Request JSON mode if supported by your provider")
    parser.add_argument("--dry-run", action="store_true", help="Print plan statistics and first request; no API/files")
    parser.add_argument("--resume", action="store_true",
                        help="Append --count new samples to an existing output and manifest")
    args = parser.parse_args(argv)
    for key in ("count", "batch_size", "max_tokens", "max_assistant_chars", "max_asr_chars"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 <= args.temperature <= 2 or not 0 < args.timeout < float("inf"):
        parser.error("temperature must be in [0,2] and timeout must be finite and positive")
    if args.batch_size < 3:
        parser.error("--batch-size must be at least 3 to keep topic contrasts together")
    if args.topic is not None and not args.topic.strip():
        parser.error("--topic must not be empty")
    if args.max_requests is None:
        effective_batch = args.batch_size - args.batch_size % 3
        args.max_requests = 3 * ((args.count + effective_batch - 1) // effective_batch)
    if args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    if not args.dry_run and not args.model:
        parser.error("--model is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        plan = make_plan(args.count, args.seed, args.topic, args.max_asr_chars, args.max_assistant_chars,
                         args.special_cases)
        print(json.dumps({"count": len(plan), "labels": dict(Counter(x["label"] for x in plan)),
                          "length_hints": dict(Counter(x["length_hint"] for x in plan)),
                          "assistant_contexts": dict(Counter(x["assistant_context"] for x in plan)),
                          "max_requests": args.max_requests,
                          "first_request": make_messages(select_batch(plan, args.batch_size), [],
                                                         args.max_assistant_chars, args.max_asr_chars)},
                         ensure_ascii=False, indent=2))
        return 0
    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"Missing API key environment variable: {args.api_key_env}", file=sys.stderr)
        return 1
    manifest = args.output.with_name(args.output.name + ".manifest.jsonl")
    if not args.output.parent.is_dir():
        print("Output parent must already exist", file=sys.stderr)
        return 1
    try:
        if args.resume:
            if not args.output.exists() or not manifest.exists():
                print("--resume requires both the existing output and manifest", file=sys.stderr)
                return 1
            initial_rows = load_existing_data(args.output)
            data_mode = "a"
            manifest_mode = "a"
        else:
            if args.output.exists() or manifest.exists():
                print("Output/manifest already exists; use --resume to append", file=sys.stderr)
                return 1
            initial_rows = []
            data_mode = "x"
            manifest_mode = "x"
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the project's declared openai dependency before generation", file=sys.stderr)
        return 1
    try:
        with OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout, max_retries=0) as client:
            with args.output.open(data_mode, encoding="utf-8") as data_file:
                with manifest.open(manifest_mode, encoding="utf-8") as manifest_file:
                    return generate(client, args, data_file, manifest_file, initial_rows)
    except KeyboardInterrupt:
        print("Interrupted; partial files retained. Use a new output path for another run.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"File operation failed: {type(exc).__name__}; inspect partial files before retrying", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
