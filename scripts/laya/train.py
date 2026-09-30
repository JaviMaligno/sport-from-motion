"""Fine-tune Laya on one (condition, fold, seed) and predict the held-out fold.

Follows the official 2xT4 notebook (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
in NandhaKishorM/laya): same loss (Gaussian-perturbation policy gradient on a proper scoring
reward + soft cross-entropy), same optimiser, learning rates, schedule, exploration noise,
epochs, effective batch of 64 sequences, and the same post-training temperature fit on a
slice held out of training. The one change is the device layout: each job runs on a single
GPU with twice the gradient accumulation (8 x 8 instead of 8 x 2 GPUs x 4), so the two T4s
train two folds at once instead of sharing one.

Nothing is tuned on the test fold: hyperparameters are the notebook's, the number of epochs
is fixed, and the temperature is fitted on training clips only.

    python train.py --items items.jsonl --model-dir <laya>/multilingual --out out \
        --jobs motion:0:0,motion:1:0 --device cuda:0
    python train.py ... --zero-shot       # the untuned checkpoint, same scoring path
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 8          # 8 x 8 = 64 sequences per update, as 8 x 2 GPUs x 4 in the notebook
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1
CALIB_FRACTION, CALIB_MAX = 10, 400
CONTROL_TAGS = {"soccer": "Q7", "basketball": "M2", "american_football": "Z9", "handball": "R4"}


def make_item(tok, cfg, row, control=False):
    """`control` prepends an arbitrary tag that encodes the answer: a positive control of the
    training code, not a condition. The tag carries no meaning, so the untuned model cannot
    read it; only a model that learns the association during fine-tuning can use it."""
    if control:
        row = {**row, "state": f"tag: {CONTROL_TAGS[row['sport']]}\n" + row["state"]}
    crit = row["criteria"]
    keys = list(crit)
    q = {"t": "choice", "ins": row["instructions"], "crit": crit}
    seq, markers, trunc = build_sequence(tok, row["state"], q, cfg["max_len"], cfg["head_max_len"],
                                         return_truncation_stats=True)
    if len(markers) != len(render_options(q)):
        raise ValueError(f"marker mismatch on {row['item_id']}")
    target = [1.0 if k == row["sport"] else 0.0 for k in keys]
    return {"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": target,
            "label": target.index(1.0), "keys": keys, "row": row, "truncated": trunc["truncated"],
            "n_tokens": len(seq)}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def forward(model, batch, device, amp):
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                            batch["qtype"].to(device))
    return logits.float(), act


@torch.no_grad()
def raw_logits(model, items, pad_id, device, amp):
    model.eval()
    out = []
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        logits, _ = forward(model, collate(chunk, pad_id), device, amp)
        l = logits.cpu().numpy()
        out += [l[r, :len(it["markers"])] for r, it in enumerate(chunk)]
    return out


def fit_temperature(logits, items):
    """The notebook's fit_one_temp: one scalar, LBFGS on held-out log loss, clamped to [0.1, 10]."""
    if len(items) < 10:
        return 1.0
    Z = torch.tensor(np.stack(logits))
    T = torch.tensor([it["target"] for it in items])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def load_base(model_dir, device):
    _fix_tokenizer_config(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return cfg, tok, model.to(device)


def train(model, items, tok, device, amp, seed, max_steps=None, log=print, epochs=EPOCHS,
          schedule="cosine", on_epoch=None):
    """`schedule="cosine"` is the notebook's. `"constant"` (DL1) keeps the initial learning
    rates and anneals sigma over the notebook's first 4 epochs only, so the model after epoch
    E does not depend on the total and one long run yields every shorter budget."""
    torch.manual_seed(seed)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}],
                            weight_decay=0.01)
    total = max(1, (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * epochs)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
             if schedule == "cosine" else torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0))
    scaler = (torch.amp.GradScaler(device.type, enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))
    order = list(items)
    steps = 0
    for epoch in range(epochs):
        random.Random(seed * 1000 + epoch).shuffle(order)
        span = epochs if schedule == "cosine" else EPOCHS
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * min(1.0, epoch / max(1, span - 1))
        opt.zero_grad(set_to_none=True)
        tot, tot_ce, nb = 0.0, 0.0, 0
        for b in range(0, len(order), MICRO_BATCH):
            batch = collate(order[b:b + MICRO_BATCH], tok.pad_token_id)
            logits, act = forward(model, batch, device, amp)
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            scaler.scale(loss).backward()
            nb += 1
            tot += loss.item() * GRAD_ACCUM
            tot_ce += loss_ce.item()
            if nb % GRAD_ACCUM == 0 or b + MICRO_BATCH >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                if max_steps and steps >= max_steps:
                    log(f"  smoke stop after {steps} updates, loss {tot / nb:.4f}")
                    return
        log(f"  epoch {epoch + 1}/{epochs} loss {tot / max(1, nb):.4f} ce {tot_ce / max(1, nb):.4f} updates {steps}")
        if on_epoch:
            on_epoch(epoch + 1)
            model.train()


def softmax(z, t):
    e = np.exp((z - z.max()) / t)
    return e / e.sum()


def run_job(args, rows, cond, fold, seed, device, amp):
    t0 = time.time()
    cfg, tok, model = load_base(args.model_dir, device)
    pool = [r for r in rows if r["condition"] == cond]
    test = [make_item(tok, cfg, r, args.control) for r in pool if r["fold"] == fold]
    name = "laya-zs" if args.zero_shot else "laya-ft"
    stem = "laya-zs" if args.zero_shot else f"laya-ft-s{seed}"
    if args.tag and not args.zero_shot:
        name, stem = f"{name}-{args.tag}", stem.replace("laya-ft-", f"laya-ft-{args.tag}-")
    if args.control:
        name, stem = name + "-control", stem.replace("laya-", "laya-control-")
    info = {"model": name, "condition": cond, "fold": fold, "seed": seed, "n_test": len(test)}
    if args.zero_shot:
        temp = float(cfg["temperature"][QTYPES["choice"]])
    else:
        train_items = [make_item(tok, cfg, r, args.control) for r in pool if r["fold"] != fold]
        idx = list(range(len(train_items)))
        random.Random(20260922 + seed).shuffle(idx)
        n_cal = min(CALIB_MAX, len(train_items) // CALIB_FRACTION)
        calib = [train_items[i] for i in sorted(idx[:n_cal])]
        fit = [train_items[i] for i in sorted(idx[n_cal:])]
        if args.smoke:
            fit, calib, test = fit[:32], calib[:16], test[:16]
        curve = []

        def on_epoch(e):
            if args.eval_each_epoch:
                acc = float(np.mean([int(np.argmax(z)) == it["label"] for it, z in
                                     zip(calib, raw_logits(model, calib, tok.pad_token_id, device, amp))]))
                curve.append({"epoch": e, "calib_accuracy": acc})
                print(f"  epoch {e} calib accuracy {acc:.3f}", flush=True)

        train(model, fit, tok, device, amp, seed, max_steps=2 if args.smoke else None,
              epochs=args.epochs, schedule=args.schedule, on_epoch=on_epoch)
        info.update(epochs=args.epochs, schedule=args.schedule, calib_curve=curve)
        temp = fit_temperature(raw_logits(model, calib, tok.pad_token_id, device, amp), calib)
        info.update(n_train=len(fit), n_calib=len(calib),
                    truncated_train=sum(it["truncated"] for it in train_items))
    logits = raw_logits(model, test, tok.pad_token_id, device, amp)
    info.update(truncated_test=sum(it["truncated"] for it in test),
                max_tokens=max(it["n_tokens"] for it in test))
    out = os.path.join(args.out, f"{stem}__{cond}__fold{fold}.jsonl")
    with open(out, "w") as f:
        for it, z in zip(test, logits):
            p = softmax(z, temp)
            row = it["row"]
            probs = {k: float(v) for k, v in zip(it["keys"], p)}
            f.write(json.dumps({
                "item_id": row["item_id"], "clip_id": row["clip_id"], "sport": row["sport"],
                "match_id": row["match_id"], "model": name, "condition": cond, "repr": "text",
                "prompt_style": "neutral", "replicate": 1 if args.zero_shot else seed + 1, "fold": fold, "eval400": row["eval400"],
                "label": max(probs, key=probs.get), "probs": probs, "logits": [float(x) for x in z],
                "temperature": temp, "error": None}) + "\n")
    info.update(temperature=temp, seconds=round(time.time() - t0, 1),
                accuracy=float(np.mean([max(zip(z, it["keys"]))[1] == it["row"]["sport"] for it, z in zip(test, logits)])))
    with open(out.replace(".jsonl", ".info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True, help="condition:fold:seed, comma-separated")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--eval-each-epoch", action="store_true", help="calibration-slice accuracy per epoch (DL1)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. e16")
    ap.add_argument("--control", action="store_true", help="positive control: the answer is in the state")
    ap.add_argument("--smoke", action="store_true", help="2 updates on 32 clips, to test the code")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.items)]
    device = torch.device(args.device)
    amp = device.type == "cuda"
    for job in args.jobs.split(","):
        cond, fold, seed = job.split(":")
        stem = "laya-zs" if args.zero_shot else "laya-ft-s" + seed
        if args.tag and not args.zero_shot:
            stem = stem.replace("laya-ft-", f"laya-ft-{args.tag}-")
        if args.control:
            stem = stem.replace("laya-", "laya-control-")
        done = os.path.join(args.out, f"{stem}__{cond}__fold{fold}.info.json")
        if os.path.exists(done):
            print(f"skip {job}: already done", flush=True)
            continue
        run_job(args, rows, cond, int(fold), int(seed), device, amp)


if __name__ == "__main__":
    main()
