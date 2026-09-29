"""Build a fingerprinted frozen-model probability cache, using training data only."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import rlcd_taxonomies as TAX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preserve-first', type=int, required=True,
                        help='Original rows to preserve; subsequent augmentation rows get no KL')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not 0 <= args.preserve_first <= len(rows):
        raise ValueError('preserve-first outside dataset range')
    config_path = Path(args.model) / 'rlcd_config.json'
    if not config_path.is_file():
        raise ValueError('Reference must be a local checkpoint with rlcd_config.json')
    tax = json.loads(config_path.read_text())
    expected = TAX.get('continue')
    if any(tax[k] != expected[k] for k in ['labels', 'system', 'user_tmpl']):
        raise ValueError('Reference prompt/labels differ from training snapshot')
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()
    slots = [tok.encode(k, add_special_tokens=False) for k in ['A', 'B', 'C']]
    if not all(len(x) == 1 for x in slots):
        raise ValueError('Expected single-token options')
    slots = [x[0] for x in slots]
    probabilities = []
    with torch.inference_mode():
        for i in range(0, args.preserve_first, 2):
            items = []
            for row in rows[i:min(i + 2, args.preserve_first)]:
                messages = [{'role': 'system', 'content': tax['system']}, {'role': 'user', 'content': tax['user_tmpl'].format(assistant=row['assistant_text'], user=row['user_asr'])}]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                items.append(tok.encode(prompt, add_special_tokens=False))
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            ids = torch.full((len(items), max(map(len, items))), pad, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(ids)
            for j, ids_row in enumerate(items):
                ids[j, :len(ids_row)] = torch.tensor(ids_row, device='cuda')
                mask[j, :len(ids_row)] = 1
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            probabilities.extend(logits[torch.arange(len(items), device='cuda'), mask.sum(1)-1][:, slots].float().softmax(-1).cpu().tolist())
            if i % 1000 == 0:
                print(f'reference {i}/{args.preserve_first}', flush=True)
    probabilities += [None] * (len(rows) - args.preserve_first)
    result = {'checkpoint': args.model, 'labels': tax['labels'], 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(), 'probabilities': probabilities}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
