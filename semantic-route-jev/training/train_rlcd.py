"""V7-B training: Gaussian-logit REINFORCE/RLOO + soft CE + masked reference KL.
Project implementation trained with Jev-derived targets; not official Jev source.
Use train.sh for the recorded V7-B hyperparameters. CUDA required.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from preservation import load_reference, preservation_kl
from rlcd_algorithm import sigma_at_step, advantages, logit_gradient_stats

import sys as _sys
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in _sys.path:
    _sys.path.insert(0, _BASE)
import rlcd_taxonomies as TAX  # noqa

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

APPEND_A = os.environ.get("APPEND_A", "0") == "1"          # 1 = 旧行为（多喂一个 "A"）
TAKE_REAL_LEN = os.environ.get("TAKE_REAL_LEN", "1") == "1" # 0 = 旧行为（盲取 [:, -1]）
TRAIN_MODE = ("legacy" if (APPEND_A and not TAKE_REAL_LEN) else
              "padfix_only" if (APPEND_A and TAKE_REAL_LEN) else
              "posfix_only" if (not APPEND_A and not TAKE_REAL_LEN) else
              "fixed")   # APPEND_A=0 && TAKE_REAL_LEN=1

TAXONOMY = TAX.get(os.environ.get("TAXONOMY", "continue"))
LABELS = TAXONOMY["labels_upper"]
LABEL_KEYS = TAXONOMY["labels"]
SYSTEM = TAXONOMY["system"]
USER_TMPL = TAXONOMY["user_tmpl"]


def proper_reward(q, target, w_sph=0.75, log_floor=None):
    """Target-weighted log and spherical reward for sampled distributions."""
    logq = torch.log(q.clamp_min(1e-12))
    if log_floor is not None:
        logq = logq.clamp_min(log_floor)
    log_score = (target.unsqueeze(0) * logq).sum(-1)
    sph = (target.unsqueeze(0) * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    return log_score + w_sph * sph


def main():
    SEED = int(os.environ.get("SEED", "0"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    DETERMINISTIC = os.environ.get("DETERMINISTIC", "0") == "1"
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子: {SEED}  严格确定性: {DETERMINISTIC}")
    assert torch.cuda.is_available(), "CUDA is required"

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.45)
    data_path = os.environ["DATA"]
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    print(f"数据: {len(rows)} 条  ({data_path})")
    print(f"体系: {TAXONOMY['name']} ({'/'.join(LABEL_KEYS)})  —— {TAXONOMY['title']}")
    print(f"取位模式: {TRAIN_MODE}  (APPEND_A={int(APPEND_A)} TAKE_REAL_LEN={int(TAKE_REAL_LEN)})")
    if TRAIN_MODE == "fixed":
        print("  → 训练读的分布与 serve_rlcd.py 一致（prompt 末位、按真实长度）")
    elif TRAIN_MODE == "legacy":
        print("  → 复现旧行为：追加 'A' 后盲取 [:, -1]（取位 + padding 两个 bug 都在）")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    print("FP32 trainable parameters and optimizer state; BF16 autocast forward", flush=True)

    keys = ["A", "B", "C"]
    slot_ids = []
    for k in keys:
        ids = tok.encode(k, add_special_tokens=False)
        assert len(ids) == 1, f"'{k}' 不是单 token: {ids}"
        slot_ids.append(ids[0])
    assert len(set(slot_ids)) == 3, "slot 冲突"
    print(f"选项 token: A={slot_ids[0]} B={slot_ids[1]} C={slot_ids[2]}")

    def build(row):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(
                    assistant=row["assistant_text"], user=row["user_asr"])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        ids = tok.encode(prompt, add_special_tokens=False)
        for i, sid in enumerate(slot_ids):
            assert tok.encode(prompt + keys[i], add_special_tokens=False) == ids + [sid], \
                f"选项边界改变了 tokenization: {keys[i]}"
        tail = ids + [slot_ids[0]] if APPEND_A else ids
        return ids, tail, np.array([row["target"][k] for k in LABEL_KEYS], dtype=np.float32)

    ref_weight = float(os.environ.get("REFERENCE_KL_WEIGHT", "2.0"))
    references, reference_masks = load_reference(os.environ["REFERENCE_FILE"], data_path, len(rows))
    assert ref_weight >= 0
    if not rows:
        raise ValueError("Empty training dataset")
    if LABEL_KEYS != ["continue", "yield", "wait"]:
        raise ValueError("V7-B reference cache supports continue/yield/wait only")
    for row in rows:
        if not isinstance(row.get("assistant_text"), str) or not isinstance(row.get("user_asr"), str):
            raise ValueError("assistant_text and user_asr must be strings")
        target = row.get("target", {})
        if set(target) != set(LABEL_KEYS) or any(not np.isfinite(v) or not 0 <= v <= 1 for v in target.values()) or abs(sum(target.values()) - 1) > 1e-4:
            raise ValueError("target must be a normalized three-class distribution")
    items = []
    for row_index, r in enumerate(rows):
        ids, tail, tgt = build(r)
        items.append({"ids": ids, "tail": tail, "target": tgt, "len": len(ids), "reference": references[row_index], "reference_mask": reference_masks[row_index]})
    lens = np.array([it["len"] for it in items])
    print(f"prompt token: min={lens.min()} median={int(np.median(lens))} max={lens.max()}")

    EPOCHS = int(os.environ.get("EPOCHS", "6"))
    BATCH = int(os.environ.get("BATCH", "8"))
    CLASS_BALANCE = os.environ.get("CLASS_BALANCE", "0") == "1"
    BALANCE_POWER = float(os.environ.get("BALANCE_POWER", "1.0"))
    GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
    GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "4"))
    SIGMA_START, SIGMA_END = 0.4, 0.1
    ADVANTAGE_MODE = os.environ.get("ADVANTAGE_MODE", "rloo")
    assert ADVANTAGE_MODE in ("legacy", "rloo")
    optimizer_step = 0
    diagnostics_path = os.environ.get("DIAGNOSTICS_OUT", "gradient_diagnostics.jsonl")
    if os.path.exists(diagnostics_path):
        raise FileExistsError(diagnostics_path)
    LR_BACKBONE = float(os.environ.get("LR_BACKBONE", "1e-5"))
    LR_HEAD = float(os.environ.get("LR_HEAD", "1e-5"))
    W_SPH = 0.75
    CE_WEIGHT = 1.0

    def split_params():
        head, back = [], []
        head_keys = ("lm_head", "score", "norm")
        for n, p in model.named_parameters():
            (head if any(h in n for h in head_keys) else back).append(p)
        return head, back

    head_p, back_p = split_params()
    opt = torch.optim.AdamW([
        {"params": back_p, "lr": LR_BACKBONE},
        {"params": head_p, "lr": LR_HEAD},
    ], weight_decay=0.01)
    n_max = int(np.ceil(np.ceil(len(items) / BATCH) / GRAD_ACCUM)) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_max), eta_min=min(LR_BACKBONE, LR_HEAD) * 0.1)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    hist = []

    _cls = np.stack([it["target"] for it in items]).argmax(1)

    TARGET_PRIOR = os.environ.get("TARGET_PRIOR", "")
    prior_p = None
    if TARGET_PRIOR:
        want = np.array([float(x) for x in TARGET_PRIOR.split(",")], dtype=np.float64)
        assert len(want) == len(LABEL_KEYS), f"TARGET_PRIOR 需要 {len(LABEL_KEYS)} 个数"
        assert want.sum() > 0
        want = want / want.sum()
        cur = np.bincount(_cls, minlength=len(LABEL_KEYS)).astype(np.float64) / len(_cls)
        w_cls = np.where(cur > 0, want / np.clip(cur, 1e-12, None), 0.0)
        sw = w_cls[_cls]
        prior_p = sw / sw.sum()
        print("先验重采样: 开启  当前=%s  目标=%s"
              % (np.round(cur, 4).tolist(), np.round(want, 4).tolist()))

    if CLASS_BALANCE:
        freq = np.bincount(_cls, minlength=len(LABEL_KEYS)).astype(np.float64)
        w = (1.0 / np.clip(freq, 1, None)) ** BALANCE_POWER
        sample_w = w[_cls]
        sample_p = sample_w / sample_w.sum()
        print("类别平衡: 开启 (power=%.2f)  freq=%s  -> 采样权重=%s"
              % (BALANCE_POWER, freq.astype(int).tolist(), np.round(w, 4).tolist()))

    for epoch in range(EPOCHS):
        if prior_p is not None:
            idxs = np.random.choice(len(items), size=len(items), replace=True, p=prior_p)
            epoch_items = [items[i] for i in idxs]
        elif CLASS_BALANCE:
            idxs = np.random.choice(len(items), size=len(items), replace=True, p=sample_p)
            epoch_items = [items[i] for i in idxs]
        else:
            random.shuffle(items)
            epoch_items = items
        items_ep = epoch_items
        sigma = sigma_at_step(optimizer_step, n_max, SIGMA_START, SIGMA_END)
        ep_loss, ep_rew, nb, accum = 0.0, 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        for bi in range(0, len(items_ep), BATCH):
            sigma = sigma_at_step(optimizer_step, n_max, SIGMA_START, SIGMA_END)
            chunk = items_ep[bi:bi + BATCH]
            L = max(len(it["tail"]) for it in chunk)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, it in enumerate(chunk):
                ids[i, :len(it["tail"])] = torch.tensor(it["tail"])
                att[i, :len(it["tail"])] = 1
            target = torch.tensor(np.stack([it["target"] for it in chunk])).to("cuda")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.to("cuda"), attention_mask=att.to("cuda"))
                if TAKE_REAL_LEN:
                    li = (att.sum(-1) - 1).clamp(min=0)
                    rows_ix = torch.arange(out.logits.size(0), device=out.logits.device)
                    logits = out.logits[rows_ix, li][:, slot_ids].float()
                else:
                    logits = out.logits[:, -1, :][:, slot_ids].float()

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=logits.device) * sigma
            eps = eps - eps.mean(-1, keepdim=True)          # 零均值投影
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z, -1)

            with torch.no_grad():
                r = proper_reward(q, target, w_sph=W_SPH)
                adv = advantages(r, ADVANTAGE_MODE)

            logp = -((z - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
            reference = torch.tensor(np.stack([it["reference"] for it in chunk]), device=logits.device)
            reference_mask = torch.tensor([it["reference_mask"] for it in chunk], device=logits.device)
            loss_keep = preservation_kl(logits, reference, reference_mask)
            window_start = (bi // BATCH // GRAD_ACCUM) * GRAD_ACCUM
            window_size = min(GRAD_ACCUM, int(np.ceil(len(items_ep) / BATCH)) - window_start)
            loss = (loss_rl + CE_WEIGHT * loss_ce + ref_weight * loss_keep) / window_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            if nb == 0 or (nb + 1) % 100 == 0:
                diagnostics = logit_gradient_stats(logits, {"rl": loss_rl, "ce": CE_WEIGHT * loss_ce, "kl": ref_weight * loss_keep})
                diagnostics.update(epoch=epoch+1, microbatch=nb+1, optimizer_step=optimizer_step, sigma=sigma,
                    advantage_mode=ADVANTAGE_MODE, reward_std=float(r.std()), advantage_std=float(adv.std()),
                    rl_loss=float(loss_rl.detach()), ce_loss=float(loss_ce.detach()), keep_kl=float(loss_keep.detach()),
                    note="Weighted loss gradients at logits, before accumulation and parameter clipping; not full parameter gradient norms")
                with open(diagnostics_path,"a") as handle:
                    handle.write(json.dumps(diagnostics)+"\n")
            loss.backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or bi + BATCH >= len(items_ep):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                optimizer_step += 1
                sched.step()
                opt.zero_grad(set_to_none=True)

            ep_loss += loss.item() * window_size
            ep_rew += r.mean().item()
            nb += 1
            if nb == 1 or nb % 100 == 0:
                print(f"progress epoch={epoch+1} batch={nb}/{int(np.ceil(len(items_ep)/BATCH))} loss={loss.item()*window_size:.4f} keep_kl={loss_keep.item():.5f} sigma={sigma:.5f} step={optimizer_step}/{n_max} lr={opt.param_groups[0]['lr']:.3g} allocated_GB={torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)

        _ep_cls = np.stack([it["target"] for it in items_ep]).argmax(1)
        _ep_dist = {k: int((_ep_cls == i).sum()) for i, k in enumerate(LABEL_KEYS)}
        hist.append({"epoch": epoch + 1, "loss": ep_loss / nb, "reward": ep_rew / nb,
                     "sigma": sigma, "sec": time.time() - t0, "class_dist": _ep_dist})
        print(f"epoch {epoch+1}/{EPOCHS} | loss {ep_loss/nb:.4f} | reward {ep_rew/nb:.4f} "
              f"| sigma {sigma:.2f} | {time.time()-t0:.1f}s", flush=True)

    OUT = os.environ["CKPT_OUT"]
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    cfg_path = TAX.save(OUT, TAXONOMY["name"], extra={
        "base_model": MODEL_ID,
        "epochs": EPOCHS,
        "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
        "objective": "REINFORCE + soft CE(1) + masked KL(V4 || student)",
        "advantage_mode": ADVANTAGE_MODE,
        "sigma_schedule": "linear_by_optimizer_update",
        "optimizer_updates": optimizer_step,
        "reference_kl_weight": ref_weight,
        "reference_file": os.environ["REFERENCE_FILE"],
        "reference_sha256": __import__("hashlib").sha256(open(os.environ["REFERENCE_FILE"],"rb").read()).hexdigest(),
        "data_sha256": __import__("hashlib").sha256(open(data_path,"rb").read()).hexdigest(),
        "train_rows": len(items),
        "train_mode": TRAIN_MODE,
        "append_a": APPEND_A,
        "take_real_len": TAKE_REAL_LEN,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "target_prior": TARGET_PRIOR,
        "gold_distribution": {k: int((np.stack([it["target"] for it in items]).argmax(1) == i).sum())
                              for i, k in enumerate(LABEL_KEYS)},
    })
    print("已写入体系配置 → %s (taxonomy=%s, train_mode=%s)"
          % (cfg_path, TAXONOMY["name"], TRAIN_MODE))
    json.dump(hist, open(os.environ.get("HIST_OUT", os.path.join(OUT, "train_history.json")), "w"), indent=2)
    print("已保存 → %s" % OUT)


if __name__ == "__main__":
    main()
