#!/usr/bin/env python
"""Offline answer-token soft-KD for zagreus-0.4B on ITALIC MCQ (family B, NEW variant).

Classic Hinton "dark-knowledge" distillation on JUST the answer token, from the
STRONGEST teacher (Mistral-Small-3.2-24B, precomputed soft labels). This is NOT
sequence-KD and NOT on-policy GKD: no teacher at train time, no generation. For
each MCQ we already have the teacher's first-token logprobs over the option
letters (byte-identical 5-shot-fast eval format, temp 0). The loss at the single
answer position is:

    L = T^2 * CE(student_letter_dist@T,  teacher_soft@T)   +   lambda * CE(gold)

- KD term: cross-entropy between the student's next-token distribution restricted
  & renormalized over the option-letter token ids and the teacher soft dist
  q = softmax(teacher_logprob / T). Gradient-equivalent to KL(q || student), T^2
  scaled (Hinton). This is the richest signal the harness can use: the teacher's
  split of mass over distractors ("B right, A a plausible trap").
- CE term: standard hard cross-entropy on the gold-letter token over the full
  vocab, weight lambda. Anchors toward gold and suppresses non-letter emission.

Why this is new vs the prior no-op/NaN distills:
  (1) SEED = r2union CONTENT checkpoint (full-10K 0.2876, per-gold>emission on
      every letter = real content), not base zagreus / content-less SFT.
  (2) STABLE: fp32 student (bf16 AdamW updates fall below ULP at 0.4B scale -> no-op),
      max_grad_norm=1.0, warmup, skip-non-finite guards. bf16-no-clip was what
      NaN'd/no-op'd every prior distill run.
  (3) TRAIN==TEACHER==EVAL prompt: always 5-shot (the teacher was scored 5-shot;
      the eval is always 5-shot). The prior train_distill trained 0-shot -> mismatch.
  (4) NO FIXED LENGTH: dev500 best-ckpt + early-stop, train long. RMS(dW) logged to
      confirm weights move above the ~2e-4 bf16-ULP floor.

Answer position: the prompt ends '<|im_start|>assistant\\n' (add_generation_prompt);
the logits at the LAST prompt token predict the answer letter -- exactly the token
extract_answer_fast parses. Right-padded batch; per-row answer_pos = len(prompt)-1.

Usage (GSC1 vllm-venv):
  python train_softkd.py --data data/sft/Bsoftkl.jsonl --tag mistral-softkd \
    --student experiments/undertrain-r2union/model --shots eval/5_shots.jsonl \
    --dev-file data/italic_dev500.jsonl --temp 2.0 --lmbda-ce 0.5 \
    --lr 1e-4 --bs 8 --grad-accum 4 --eval-steps 100 --patience 8 --max-steps 2000
"""
import argparse, json, math, os, random, re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SYSTEM = "Sei un assistente utile."
LETTER = re.compile(r"[A-Z]")  # == eval/run_eval.py extract_answer_fast
FAST_TMPL = (
    "Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. "
    "La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) "
    "dove LETTERA è una tra {merged}. Scrivi solo la lettera corrispondente alla tua "
    "risposta senza spiegazioni.\n\n{question}\n\n{options}\n\nRisposta:")


def fmt_opts(options):
    lines = "\n".join(f"{list(o.keys())[0]}) {list(o.values())[0]}" for o in options)
    letters = "".join(list(o.keys())[0] for o in options)
    return lines, letters


def user_msg(it):
    opts, letters = fmt_opts(it["options"])
    return FAST_TMPL.format(topic=it.get("category", ""), merged=letters,
                            question=it["question"], options=opts)


def build_shot_turns(path):
    turns = []
    for line in open(path):
        it = json.loads(line)
        turns.append({"role": "user", "content": user_msg(it)})
        turns.append({"role": "assistant", "content": it["answer"]})
    return turns


def answer_token_id(tok, letter):
    """The single token id the model emits for `letter` right after 'assistant\\n'."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": letter}]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template(msgs, add_generation_prompt=False, return_dict=False)
    comp = full[len(prompt):]
    # transformers-5.14.1 TokenizersBackend returns '' for isolated single-token
    # decode → match on the raw BPE piece (strip Llama-3 space/newline markers).
    for t in comp:
        piece = tok.convert_ids_to_tokens(t).replace("Ġ", "").replace("Ċ", "").strip()
        if piece == letter:
            return t
    raise AssertionError(f"no answer token for {letter!r}: comp={comp} "
                         f"pieces={tok.convert_ids_to_tokens(comp)}")


class SoftKDData(Dataset):
    """5-shot MCQ prompt (ends assistant\\n) + soft target over option-letter ids + gold id."""
    def __init__(self, path, tok, shot_turns, letter2id, max_len, temp, limit=None):
        self.ex, skipped = [], 0
        for line in open(path):
            it = json.loads(line)
            msgs = [{"role": "system", "content": SYSTEM}] + shot_turns + \
                   [{"role": "user", "content": user_msg(it)}]
            prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False)
            if len(prompt_ids) > max_len:
                skipped += 1
                continue
            letters = list(it["teacher_logprobs"].keys())
            lps = torch.tensor([it["teacher_logprobs"][L] for L in letters])
            target = F.softmax(lps / temp, dim=0)          # teacher soft dist @T over letters
            lids = [letter2id[L] for L in letters]
            gold_id = letter2id[it["answer"]]
            self.ex.append((prompt_ids, lids, target.tolist(), gold_id))
            if limit and len(self.ex) >= limit:
                break
        print(f"[softkd] loaded {len(self.ex)} examples (skipped {skipped} > max_len={max_len}), "
              f"temp={temp}", flush=True)

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    m = max(len(b[0]) for b in batch)
    kmax = max(len(b[1]) for b in batch)
    ids, mask, pos, lids, targ, gold = [], [], [], [], [], []
    for prompt_ids, letter_ids, target, gold_id in batch:
        n = m - len(prompt_ids)
        ids.append(prompt_ids + [pad_id] * n)             # right-pad: position_ids=arange is correct
        mask.append([1] * len(prompt_ids) + [0] * n)
        pos.append(len(prompt_ids) - 1)                   # last real token predicts the answer letter
        k = kmax - len(letter_ids)
        lids.append(letter_ids + [0] * k)                 # pad slot id 0, masked by target==0
        targ.append(target + [0.0] * k)
        gold.append(gold_id)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
            "answer_pos": torch.tensor(pos), "letter_ids": torch.tensor(lids),
            "target": torch.tensor(targ), "gold_ids": torch.tensor(gold)}


def softkd_loss(logits, batch, temp, lmbda_ce):
    b = torch.arange(logits.size(0), device=logits.device)
    ans = logits[b, batch["answer_pos"]].float()          # [B, V] logits at answer position
    # --- KD term over the option letters (Hinton dark knowledge) ---
    letter_logits = torch.gather(ans, 1, batch["letter_ids"])   # [B, Kmax]
    pad = batch["target"] == 0
    letter_logits = letter_logits.masked_fill(pad, float("-inf"))
    logq = F.log_softmax(letter_logits / temp, dim=1)     # student log-dist over letters @T
    kd = -(batch["target"] * logq.nan_to_num(0.0)).sum(1).mean() * (temp ** 2)
    # --- CE term on the hard gold letter over the full vocab ---
    ce = F.cross_entropy(ans, batch["gold_ids"])
    return kd + lmbda_ce * ce, kd.detach(), ce.detach()


def param_rms(model):
    with torch.no_grad():
        sq = sum(p.double().pow(2).sum() for p in model.parameters())
        n = sum(p.numel() for p in model.parameters())
    return (sq / n).sqrt().item()


def delta_rms(model, snapshot):
    with torch.no_grad():
        sq = n = 0.0
        for p, s in zip(model.parameters(), snapshot):
            sq += (p.detach().double() - s.to(p.device)).pow(2).sum().item()
            n += p.numel()
    return (sq / n) ** 0.5


def left_pad(prompts, pad_id):
    m = max(len(p) for p in prompts)
    ids = [[pad_id] * (m - len(p)) + p for p in prompts]
    attn = [[0] * (m - len(p)) + [1] * len(p) for p in prompts]
    return torch.tensor(ids), torch.tensor(attn)


@torch.no_grad()
def dev_eval(model, tok, dev_items, shot_turns, bs, pad_id, max_new=4):
    """Greedy first-letter accuracy, 5-shot, byte-identical to the official eval.
    Also returns the prediction distribution (collapse guard)."""
    was = model.training
    model.eval()
    dev = model.device
    correct = 0
    pred_dist = {}
    for i in range(0, len(dev_items), bs):
        batch = dev_items[i:i + bs]
        prompts = []
        for it in batch:
            msgs = [{"role": "system", "content": SYSTEM}] + shot_turns + \
                   [{"role": "user", "content": user_msg(it)}]
            prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=False))
        prompts = [list(p) for p in prompts]
        ids, attn = left_pad(prompts, pad_id)
        ids, attn = ids.to(dev), attn.to(dev)
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=pad_id)
        for t, it in zip(tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True), batch):
            hit = LETTER.search(t)
            pred = hit.group(0) if hit else "<none>"
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
            correct += int(pred == it["answer"])
    if was:
        model.train()
    return correct / len(dev_items), pred_dist


def save_best(model, tok, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    # belt-and-suspenders template embed for vLLM on any transformers build (Directive-1)
    if tok.chat_template:
        open(f"{model_dir}/chat_template.jinja", "w").write(tok.chat_template)
        tcf = f"{model_dir}/tokenizer_config.json"
        tc = json.load(open(tcf))
        tc["chat_template"] = tok.chat_template
        json.dump(tc, open(tcf, "w"), ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="soft-label jsonl (teacher_logprobs per option letter)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--student", required=True, help="r2union content seed dir")
    ap.add_argument("--shots", default="eval/5_shots.jsonl")
    ap.add_argument("--dev-file", default="data/italic_dev500.jsonl")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--temp", type=float, default=2.0, help="KD temperature (Hinton)")
    ap.add_argument("--lmbda-ce", type=float, default=0.5, help="weight on hard CE(gold)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=10, help="upper bound; early-stop ends it")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="tiny dry-run subset")
    args = ap.parse_args()
    print(f"[softkd] args: {vars(args)}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = f"{args.out_root}/{args.tag}"
    model_dir = f"{out_dir}/model"
    curve_path = f"{out_dir}/dev_curve.jsonl"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(curve_path):
        os.remove(curve_path)

    tok = AutoTokenizer.from_pretrained(args.student)  # seed's own tokenizer (template embedded)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    letter2id = {L: answer_token_id(tok, L) for L in "ABCDEFGHIJ"}
    print(f"[softkd] letter->id: {dict(list(letter2id.items())[:6])} pad={pad_id}", flush=True)

    shot_turns = build_shot_turns(args.shots)
    ds = SoftKDData(args.data, tok, shot_turns, letter2id, args.max_len, args.temp, args.limit)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, pad_id))
    total = args.max_steps if args.max_steps else int(math.ceil(len(dl) / args.grad_accum) * args.epochs)

    dev_items = [json.loads(l) for l in open(args.dev_file)]
    print(f"[softkd] dev slice {args.dev_file} n={len(dev_items)}", flush=True)

    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32).cuda()
    student.config.pad_token_id = pad_id
    seed_snapshot = [p.detach().double().clone() for p in student.parameters()]
    print(f"[softkd] seed param RMS = {param_rms(student):.4f}", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "italic-challenge"),
                   entity=os.environ.get("WANDB_ENTITY", "ruggsea"), name=args.tag, config=vars(args))
    else:
        print("[softkd] no WANDB_API_KEY -> stdout only", flush=True)

    d0, pd0 = dev_eval(student, tok, dev_items, shot_turns, args.eval_bs, pad_id)
    print(f"[softkd] step0 dev500={d0:.4f} pred_dist={pd0} (seed sanity)", flush=True)
    with open(curve_path, "a") as f:
        f.write(json.dumps({"step": 0, "dev500_acc": round(d0, 4), "rms_vs_seed": 0.0,
                            "pred_dist": pd0}) + "\n")

    best, since = d0, 0
    save_best(student, tok, model_dir)  # seed is the initial best; only overwrite on gain

    student.train()
    opt_step = micro = skipped = 0
    running = rkd = rce = 0.0
    done = False
    for _ in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = student(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"]).logits
            loss, kd, ce = softkd_loss(logits, batch, args.temp, args.lmbda_ce)
            loss = loss / args.grad_accum
            del logits
            if not torch.isfinite(loss):
                skipped += 1
                print(f"[softkd] non-finite loss ({loss.item()}) -> skip micro {micro}", flush=True)
                opt.zero_grad(set_to_none=True)
                running = rkd = rce = 0.0
                continue
            loss.backward()
            running += loss.item(); rkd += kd.item() / args.grad_accum; rce += ce.item() / args.grad_accum
            micro += 1

            if micro % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    skipped += 1
                    print(f"[softkd] non-finite grad norm ({gnorm}) -> skip step {opt_step}", flush=True)
                    opt.zero_grad(set_to_none=True)
                    running = rkd = rce = 0.0
                    continue
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"[softkd] step {opt_step}/{total} loss={running:.4f} kd={rkd:.4f} "
                          f"ce={rce:.4f} lr={lr:.2e}", flush=True)
                    if use_wandb:
                        wandb.log({"loss": running, "kd": rkd, "ce": rce, "lr": lr}, step=opt_step)
                running = rkd = rce = 0.0

                if opt_step % args.eval_steps == 0:
                    acc, pdist = dev_eval(student, tok, dev_items, shot_turns, args.eval_bs, pad_id)
                    rms = delta_rms(student, seed_snapshot)
                    improved = acc > best + 1e-4
                    with open(curve_path, "a") as f:
                        f.write(json.dumps({"step": opt_step, "dev500_acc": round(acc, 4),
                                            "best": round(max(acc, best), 4),
                                            "rms_vs_seed": rms, "pred_dist": pdist}) + "\n")
                    print(f"[softkd] [deveval] step={opt_step} dev500={acc:.4f} best={max(acc, best):.4f} "
                          f"rms_vs_seed={rms:.3e} pred_dist={pdist} "
                          f"{'*SAVE*' if improved else f'no-gain {since + 1}/{args.patience}'}", flush=True)
                    if use_wandb:
                        wandb.log({"dev500_acc": acc, "rms_vs_seed": rms}, step=opt_step)
                    if improved:
                        best, since = acc, 0
                        save_best(student, tok, model_dir)
                    else:
                        since += 1
                        if since >= args.patience:
                            print(f"[softkd] early-stop: {args.patience} evals w/o gain (best={best:.4f})", flush=True)
                            done = True
                            break
                if args.max_steps and opt_step >= args.max_steps:
                    done = True
                    break

    final_rms = delta_rms(student, seed_snapshot)
    stats = {"tag": args.tag, "opt_steps": opt_step, "skipped_nonfinite": skipped,
             "best_dev500": best, "step0_dev500": d0, "rms_final_vs_seed": final_rms,
             "n_examples": len(ds), "student_seed": args.student, "data": args.data,
             "temp": args.temp, "lmbda_ce": args.lmbda_ce, "lr": args.lr}
    json.dump(stats, open(f"{out_dir}/train_stats.json", "w"), indent=2)
    print(f"[softkd] DONE best_dev500={best:.4f} step0={d0:.4f} RMS(final-seed)={final_rms:.3e} "
          f"steps={opt_step} skipped={skipped} -> {model_dir}", flush=True)


if __name__ == "__main__":
    main()
