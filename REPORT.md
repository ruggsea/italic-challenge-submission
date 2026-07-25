# ITALIC Post-Training Challenge — zagreus-0.4B → 0.4787 (report)

**Model:** mii-llm/zagreus-0.4B-ita, post-trained.
**Shipped result:** 0.4787 full-10K official ITALIC (confirmed 0.4787 / 0.4784 on independent
fresh-server runs, plus a third verification run) — the final clean-run model, one softKD run
from base, ~4.7 H100h replay. Best experimental checkpoint along the way: 0.4652
(double-confirmed 0.4652 / 0.4648). Base: 0.2802. Public leader at start: 0.372.
**+19.9pp over base, +10.7pp over the leader.**
Eval: official unmodified harness, 5-shot fast (first-capital extraction), temp 0, n=10000.

## The idea

ITALIC is 12 categories split into two kinds of question: **knowledge** (civic, tourism,
lexicon, geography, art, history, current events) and **grammar** (orthography, morphology,
syntax, synonyms, literature). The 0.4B base emits a near-constant single letter (~0.28 =
gold-A marginal): it carries almost no discriminative content, so every naive SFT/RL arm we
tried collapsed back to an emission artifact rather than learning to answer.

What actually moved the number was **soft answer-distribution knowledge distillation (softKD)
from a strong teacher committee**, staged, plus a **grammar-floor fix**:

1. **Content injection via softKD.** Instead of hard-label SFT (flat) or foreign-lineage
   sequence-KD (collapses to one letter), we distill only the teacher's probability
   distribution over the answer letters (A–J) at the answer token — cross-vocab-safe, transfers
   the teacher's per-question discrimination without importing its generation dynamics. Teacher
   = a 3-model committee (Mistral-24B + Gemma-27B + Qwen3-32B, each ≤1×H100), which measured
   0.77 on-pool, stronger than any single teacher. Iterating this from a strong seed took the
   student to **0.4387**.

2. **Grammar-floor fix (the last +2.7pp).** Per-category decomposition showed knowledge was
   near its distillation-transfer ceiling (~0.73×teacher) while grammar was the floor
   (orthography/morphology/syntax at 0.31–0.35). We committee-labeled a large Italian-grammar
   MCQ pool and **added** it to the full knowledge pool (additive, not a mix-ratio swap — an
   earlier frac-0.45 mix cut 42% of knowledge supervision and forgot knowledge for a net loss).
   The orthography/morphology categories were data-starved (only 92/563 committee-correct
   items), so we **generated + committee-labeled** more of exactly those, giving the
   `fullrich_v2` superset. Continuing softKD from the 0.4387 seed on this superset lifted every
   grammar category while knowledge held → **0.4652**.

## What worked

- **softKD (answer-distribution KL + gold CE) from a strong committee teacher** — the only
  mechanism that amplified content instead of destroying it. This is the whole spine.
- **Iterate from a strong seed**, not from base — each round re-seeds from the previous
  best-checkpoint.
- **Additive grammar supervision** (keep 100% knowledge, ADD grammar) — fixes the grammar floor
  without catastrophic forgetting. Gentler LR (1.5e-4) beats 3e-4 (less forgetting).
- **Targeted data generation** for the two starved categories (orthography, morphology).
- **Train to dev2k convergence with best-checkpoint selection + early stopping**, not a fixed
  epoch count — a previously "flat" arm gained +2.3pp just from training to convergence
  (undertraining was masking real signal).
- **Embed the chat template in every checkpoint and eval with it** (+~4pp, ships with the model,
  legal).

## What failed (and why)

- **Hard-label SFT / STaR expert-iteration** → collapse to the single most-frequent gold letter
  (emission artifact, capped ~0.29). No content.
- **On-policy GKD / off-policy sequence-KD from ANITA-8B** → foreign-lineage generation dynamics
  collapse the student to one letter (or NaN under bf16). The leader's same-lineage nesso-3B
  teacher is deleted from HF and unobtainable, so this route was structurally closed for us.
- **OPD reverse-KL at faithful low LR** → sub-bf16-ULP no-op (weights don't move); at high LR →
  degenerate collapse.
- **Mix-ratio grammar blending (frac 0.45)** → lifts grammar but forgets knowledge (net -0.6pp).
  The additive fix is what corrected this.

## Result decomposition (shipped model, 0.4787)

- per-gold accuracy {A .507, B .471, C .488, D .432} with balanced predictions
  A3068/B2597/C2624/D1709 (no single-letter collapse) — genuine content, every letter far above
  its emission rate.
- per-category: civic education .565, lexicon .551, current events .511, tourism .506,
  geography .497, synonyms .494, history .482, syntax .439, art history .437, orthography .425,
  literature .412, morphology .314. Knowledge categories are near the distillation-transfer
  ceiling; morphology remains the largest residual headroom.
- (The earlier experimental checkpoint at 0.4652, referenced elsewhere in this report for the
  research narrative, had a similar profile with weaker grammar: syntax .415, orthography .387,
  morphology .357. The shipped clean-run model beats it on 9 of 12 categories.)

## Recipe chain and the final clean run

The faithful experimental chain is ~5–6 softKD rounds (base → r2union content-SFT →
mistral-softkd → self-distill → committee-iter 0.4387 → additive ortho/morph 0.4652), which
exceeds the 10-H100h replay budget. Per the compression directive, the **final clean run**
distills the *final* committee soft labels (`grammar_committee_fullrich_v2`, 21606 items)
directly from the base model in ONE softKD run — `code/final_run.sh` + `code/final_train.slurm`.
Measured replay cost ≈4.8 H100h (synthetic generation 0.25 + committee labeling 0.47 + one softKD
run 3.70 + official eval 0.28; H200 wall clock × 1.2; full itemization in the README). The from-base clean-run model scored
**0.4787 / 0.4784** on the full official 10K, beating every experimental checkpoint, and is the
shipped artifact. A 2-stage variant (content-SFT seed → softKD) reproduced 0.4685 / 0.4683 as a
fallback. In this repository the pipeline lives in code/final_run.sh + code/final_train.slurm.

## Decontamination

Every training pool passes exact + semantic (cosine ≥0.80) decontamination against
`eval/italic.jsonl`. committee_rich: 0 exact / 0 semantic, max cosine 0.7999. committee_grammar
+ generated ortho/morph: clean at build. Counts in `data/decontam_manifest.json`.
