#!/usr/bin/env python
"""Build a COMMITTEE soft-KD pool for train_softkd.py.

The mistral-softkd winner (0.3199) trained on Bsoftkl.jsonl: pinocchio-only,
teacher-correct, civic-capped 3000, single-teacher (Mistral-24B) soft answer-dist.
This mirrors it EXACTLY but swaps the label source for the 3-teacher COMMITTEE
(mistral-24b + gemma-27b + qwen3-32b) ensemble: per item, average the teachers'
per-letter probabilities, renormalize, and store log(avg_prob) as teacher_logprobs
(byte-compatible with train_softkd.py, which does softmax(teacher_logprobs/T)).

Base questions/options come from pool_mcq.jsonl (index-aligned to the committee
labels). Filter to a single --source (default pinocchio, to match Bsoftkl). Keep
teacher-correct only unless --include-imperfect. Civic-cap mirrors the winner.

Decontam (HARD RULE, zero test items): exact question+options hash AND semantic
cosine>=0.80 vs eval/italic.jsonl (multilingual sentence-transformer). pool_mcq was
already decontaminated upstream at cosine 0.80; we RE-VERIFY here and record counts.

Output schema (== Bsoftkl.jsonl): {index, question, options, answer, category, teacher_logprobs}

Usage (data venv has sentence-transformers + torch):
  CUDA_VISIBLE_DEVICES=1 data/.venv-data/bin/python scripts/build_committee_softkd_pool.py \
    --pool tracks/committee/data/pool_mcq.jsonl \
    --labels tracks/committee/labels/pool/mistral-24b.jsonl,tracks/committee/labels/pool/gemma-27b.jsonl,tracks/committee/labels/pool/qwen3-32b.jsonl \
    --eval eval/italic.jsonl --source pinocchio --civic-cap 3000 \
    --out data/pools/committee_correct.jsonl --manifest-tag committee_correct
"""
import argparse, json, math, os, random, collections, hashlib, unicodedata, re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIM_THRESHOLD = 0.80


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def opt_texts(options):
    return [list(o.values())[0] for o in options]


def norm_hash(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def exact_key(row):
    return hashlib.md5((norm_hash(row["question"]) + "|" +
                        "|".join(sorted(norm_hash(t) for t in opt_texts(row["options"])))).encode()).hexdigest()


def embed_text(row):
    return row["question"].strip() + " " + " ".join(str(t) for t in opt_texts(row["options"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tracks/committee/data/pool_mcq.jsonl")
    ap.add_argument("--labels", required=True, help="comma-separated per-teacher label jsonls (probs per letter)")
    ap.add_argument("--eval", default="eval/italic.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-tag", required=True)
    ap.add_argument("--source", default="pinocchio", help="pool 'source' filter; empty = all sources")
    ap.add_argument("--exclude-category", default="", help="comma-sep categories to drop (Bsoftkl excluded 'other')")
    ap.add_argument("--include-imperfect", action="store_true", help="also keep committee-WRONG items")
    ap.add_argument("--civic-cap", type=int, default=3000, help="<=0 disables cap")
    ap.add_argument("--floor-margin", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool = load(args.pool)                       # index-aligned list
    teacher_files = args.labels.split(",")
    teachers = [{r["index"]: r for r in load(f)} for f in teacher_files]
    print(f"[build] pool={len(pool)} teachers={len(teachers)} ({[os.path.basename(f) for f in teacher_files]})")

    excl = {c for c in args.exclude_category.split(",") if c}
    kept, seen_q = [], set()
    n_source_skip = n_cat_skip = n_no_labels = n_wrong = n_dup = 0
    for i, it in enumerate(pool):
        if args.source and it.get("source") != args.source:
            n_source_skip += 1
            continue
        if it.get("category") in excl:
            n_cat_skip += 1
            continue
        letters = [list(o.keys())[0] for o in it["options"]]
        # average per-letter probs across teachers that labeled this index
        avg = {L: 0.0 for L in letters}
        ntea = 0
        for tmap in teachers:
            r = tmap.get(i)
            if r is None:
                continue
            ntea += 1
            pr = r["probs"]
            for L in letters:
                avg[L] += pr.get(L, 0.0)
        if ntea == 0:
            n_no_labels += 1
            continue
        for L in letters:
            avg[L] /= ntea
        s = sum(avg.values())
        if s <= 0:
            n_no_labels += 1
            continue
        for L in letters:
            avg[L] /= s                          # renormalize over the option letters
        argmax = max(letters, key=lambda L: avg[L])
        correct = (argmax == it["answer"])
        if not args.include_imperfect and not correct:
            n_wrong += 1
            continue
        q = norm_hash(it["question"])
        if q in seen_q:
            n_dup += 1
            continue
        seen_q.add(q)
        # log(avg_prob) with a floor for zero-mass letters (== build_softkd_pool floor scheme)
        logs = {L: (math.log(avg[L]) if avg[L] > 0 else None) for L in letters}
        finite = [v for v in logs.values() if v is not None]
        floor = min(finite) - args.floor_margin
        teacher_lp = {L: (logs[L] if logs[L] is not None else floor) for L in letters}
        kept.append({"index": i, "question": it["question"], "options": it["options"],
                     "answer": it["answer"], "category": it["category"],
                     "teacher_logprobs": teacher_lp, "_correct": correct})
    print(f"[build] pre-decontam kept={len(kept)}  (source_skip={n_source_skip} cat_skip={n_cat_skip} "
          f"no_labels={n_no_labels} wrong={n_wrong} dup={n_dup})")

    # ---- decontam vs eval: exact + semantic (HARD RULE) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, device=device)
    italic = load(args.eval)
    italic_keys = {exact_key(r) for r in italic}
    n_exact = 0
    kept2 = []
    for r in kept:
        if exact_key(r) in italic_keys:
            n_exact += 1
            continue
        kept2.append(r)
    # semantic
    italic_emb = model.encode([embed_text(r) for r in italic], batch_size=256,
                              convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    pool_emb = model.encode([embed_text(r) for r in kept2], batch_size=256,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    I = torch.tensor(italic_emb, device=device)
    sims = np.empty(len(kept2), dtype=np.float32)
    for st in range(0, len(kept2), 4096):
        P = torch.tensor(pool_emb[st:st+4096], device=device)
        sims[st:st+4096] = (P @ I.T).max(dim=1).values.cpu().numpy()
    clean = [r for r, sm in zip(kept2, sims) if sm < SIM_THRESHOLD]
    n_semantic = int((sims >= SIM_THRESHOLD).sum())
    max_sim = float(sims.max()) if len(sims) else 0.0
    print(f"[build] decontam: exact={n_exact} semantic={n_semantic} (max_sim={max_sim:.4f}) -> {len(clean)}")
    if n_exact or n_semantic:
        print(f"[build] WARNING: {n_exact} exact + {n_semantic} semantic eval-leaks DROPPED "
              f"(upstream pool_mcq was pre-decontaminated; these should be 0 -> investigate).")

    # ---- civic cap + shuffle ----
    if args.civic_cap > 0:
        civic = [r for r in clean if r["category"] == "civic_education"]
        rest = [r for r in clean if r["category"] != "civic_education"]
        rng.shuffle(civic)
        clean = rest + civic[: args.civic_cap]
    rng.shuffle(clean)

    n_imperfect = sum(1 for r in clean if not r["_correct"])
    for r in clean:
        del r["_correct"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)

    cats = collections.Counter(r["category"] for r in clean)
    gold = collections.Counter(r["answer"] for r in clean)
    stats = {
        "out": args.out, "pool": args.pool, "labels": teacher_files, "source": args.source,
        "exclude_category": sorted(excl),
        "include_imperfect": args.include_imperfect, "civic_cap": args.civic_cap,
        "n_teachers": len(teachers), "emb_model": EMB_MODEL, "sim_threshold": SIM_THRESHOLD,
        "pre_decontam": len(kept), "removed_exact_italic": n_exact,
        "removed_semantic_italic": n_semantic, "max_sim_final": max_sim,
        "after": len(clean), "imperfect_kept": n_imperfect,
        "gold_dist": dict(sorted(gold.items())),
        "category_coverage": dict(sorted(cats.items(), key=lambda x: -x[1])),
    }
    # merge into manifest under a committee_pools key
    mpath = "data/pools/manifest.json"
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    man.setdefault("committee_pools", {})[args.manifest_tag] = stats
    with open(mpath + ".tmp", "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    os.replace(mpath + ".tmp", mpath)

    print(f"[build] wrote {len(clean)} -> {args.out}  gold={dict(sorted(gold.items()))} imperfect={n_imperfect}")
    print(f"[build] manifest[committee_pools][{args.manifest_tag}] updated")


if __name__ == "__main__":
    main()
