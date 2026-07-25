#!/usr/bin/env python
"""Build grammar_committee_fullrich_v2.jsonl: ADDITIVE pool = ALL committee_rich
knowledge + the FULL committee-grammar set INCLUDING the new ortho/morph batch.

Never cuts knowledge (all 11453 committee_rich kept -> avoids the frac0.45 -4pp
knowledge regression). Grammar portion = committee_grammar (old) + committee_orthomorph
(new), deduped by normalized question, with a synonyms+lexicon cap so the weakest
floor categories (orthography/morphology/syntax) stay visible in the grammar portion.

Both inputs are already committee-soft-labeled AND decontaminated at build time; this
only unions + caps + shuffles. Atomic write + manifest.

  build_orthomorph_fullrich.py --grammar committee_grammar.jsonl \
    --extra committee_orthomorph.jsonl --rich committee_rich.jsonl \
    --syn-lex-cap-frac 0.5 --out data/pools/grammar_committee_fullrich_v2.jsonl
"""
import argparse, json, random, collections, os, unicodedata, re

SYN_LEX = {"synonyms_and_antonyms", "lexicon"}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def norm_q(q):
    q = unicodedata.normalize("NFKD", str(q))
    q = "".join(c for c in q if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--extra", required=True, help="new committee-labeled ortho/morph pool")
    ap.add_argument("--rich", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--syn-lex-cap-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    grammar = load(args.grammar)
    extra = load(args.extra)
    rich = load(args.rich)

    # union grammar + extra, dedup by normalized question (old grammar wins on collision)
    gram, seen = [], set()
    for r in grammar + extra:
        k = norm_q(r["question"])
        if k in seen:
            continue
        seen.add(k)
        gram.append(r)
    grammar_cats = collections.Counter(r["category"] for r in gram)

    # syn+lexicon cap within the grammar portion (keep floor visible)
    syn = [r for r in gram if r["category"] in SYN_LEX]
    core = [r for r in gram if r["category"] not in SYN_LEX]      # ortho/morph/syntax
    rng.shuffle(syn)
    cap = int(args.syn_lex_cap_frac / (1 - args.syn_lex_cap_frac) * len(core)) if args.syn_lex_cap_frac < 1 else len(syn)
    gram_final = core + syn[:cap]

    # ADDITIVE: all rich + capped grammar, dedup vs rich by question (rich = knowledge, wins)
    rich_keys = {norm_q(r["question"]) for r in rich}
    gram_add = [r for r in gram_final if norm_q(r["question"]) not in rich_keys]
    mix = rich + gram_add
    rng.shuffle(mix)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".tmp", "w") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)

    cats = collections.Counter(r["category"] for r in mix)
    stats = {"out": args.out, "grammar": args.grammar, "extra": args.extra, "rich": args.rich,
             "syn_lex_cap_frac": args.syn_lex_cap_frac,
             "grammar_union_available": len(gram), "grammar_union_cats": dict(sorted(grammar_cats.items())),
             "grammar_core_floor": len(core), "grammar_syn_lex_kept": min(cap, len(syn)),
             "grammar_added_after_rich_dedup": len(gram_add),
             "rich_kept": len(rich), "total": len(mix),
             "category_coverage": dict(sorted(cats.items(), key=lambda x: -x[1]))}
    print(json.dumps(stats, indent=2))
    mpath = "data/pools/manifest.json"
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    man.setdefault("grammar_committee_fullrich_v2", {})[os.path.basename(args.out)] = stats
    with open(mpath + ".tmp", "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    os.replace(mpath + ".tmp", mpath)
    print(f"[fullrich_v2] wrote {len(mix)} -> {args.out}  (rich {len(rich)} + grammar {len(gram_add)})")


if __name__ == "__main__":
    main()
