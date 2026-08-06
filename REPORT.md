# ITALIC Post-Training Challenge — Final Report

## Result (updated 2026-08-05 ~22:20Z)

**Final model: extkd-s1500 — 0.4932 / 0.4929 / 0.4921 full-10K official harness** (triple-confirmed, fresh-server each run).
Official `run_eval.py` fast protocol, temp 0, max_tokens 350, 10,000/10,000 parsed.

| Model | Full-10K | Notes |
|---|---|---|
| **extkd-s1500 (SHIP)** | **0.4932 / 0.4929 / 0.4921** | M2 + Stage-3 softKD on 28,561 external in-distribution Mistral-24B soft-labeled items, step 1500; triple-confirmed |
| P0aB-s500 | 0.4900 / 0.4913 (double-confirmed) | fallback ship: from-shipped softKD on 31,765 leftover committee MCQ |
| M2 | 0.4878 / 0.4880 (double-confirmed) | previous best |
| Shipped baseline | 0.4787 | challenge submission |
| Zagreus base | 0.2802 | |

## Recipe (extkd-s1500)
1. Stage 1: committee softKD from Zagreus base (3-teacher committee: Mistral-24B + gemma-27b + Qwen3-32B, probability-averaged, agreement-filtered) -> 0.4787.
2. Stage 2 (M2): culture softKD on 22,950 culture items, public 2-teacher committee (Qwen2.5-72B-AWQ + Mistral-24B), T=2.0, lmbda_ce=0.5, LR=2e-5, seed=99 -> ckpt s2000.
3. Stage 3 (extkd): softKD from M2 s2000 on 28,561 EXTERNAL items (FinancialSupport/italic_sft 20,665 + italic_sft_ext 3,268 + quiz_militare 3,641 + pinocchio sample; 0% test overlap verified exact-hash + cosine>=0.80; answer-position-balanced), same KD hyperparams, best ckpt s1500 (dev2k-official 0.501).

## Budget (replay)
Full stack replay = ~9.5-9.85 H100h of the 10 H100h cap (teacher inference included). LEGAL. See BUDGET.md.

## Decontamination
All pools: exact-hash + cosine >=0.80 (+ TF-IDF/MinHash cross-check) vs eval/italic.jsonl. External datasets: 0% exact, 0% near-dup overlap (italic_sft/ext/quiz_militare), 0.74% near-dup in pinocchio sample (dropped).


