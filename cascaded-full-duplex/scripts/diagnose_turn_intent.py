"""Diagnostic companion to benchmark_turn_intent.py; not an accuracy benchmark."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark_turn_intent import CASES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
args = parser.parse_args()
MODEL = args.model
OUTPUT = Path(args.output)
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
    local_files_only=True
).to("cuda:0").eval()


def infer(messages, limit=32):
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=limit,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    raw = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return {"raw": raw, "generation_ms": ms, "prompt": prompt}


sanity = []
for text in ["1加1等于几？只回答数字。", "把英文cat翻译成中文，只输出译文。",
             "原样输出：继续", "用户说‘嗯嗯，你继续’，意思是让对方停止说话还是继续说话？只回答停止或继续。"]:
    result = infer([{"role": "user", "content": text}])
    sanity.append(result)
    print(json.dumps({"sanity": text, **result}, ensure_ascii=False), flush=True)

system = """判断语音助手说话时，用户插话的含义。只输出一个词：
继续：只是认同、附和，或者让助手接着讲。
打断：要求停止、纠正内容、提问、改变要求、回答助手的问题。
等待：话没说完或意图不明确，包括空文本。
注意：‘不用停’表示继续，不是打断。结合上下文判断。"""
label_map = {"继续": "CONTINUE", "打断": "YIELD", "等待": "WAIT"}
rows = []
for ident, previous, spoken, text, final, expected in CASES:
    user = f"上一轮用户：{previous}\n助手已说：{spoken}\n用户当前插话：{text!r}\n转写是否结束：{'是' if final else '否'}"
    result = infer([{"role": "system", "content": system},
                    {"role": "user", "content": user}], limit=12)
    prediction = label_map.get(result["raw"], "INVALID")
    row = {"id": ident, "expected": expected, "prediction": prediction,
           "correct": prediction == expected, **result}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
report = {"model": MODEL, "dtype": args.dtype, "attention": args.attention, "note": "Prompt revision after inspecting baseline failures; same smoke cases, not held-out evaluation. Generation-only timing, not comparable to baseline full call timing.",
          "sanity": sanity, "cases": rows, "correct": sum(r["correct"] for r in rows)}
with OUTPUT.open("x") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps({"correct": report["correct"], "total": len(rows), "output": str(OUTPUT)}), flush=True)
