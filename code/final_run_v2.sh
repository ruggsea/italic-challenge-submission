#!/bin/bash
# FINAL RUN v2 — the extkd-s1500 artifact (0.4921/0.4929/0.4932 full-10K official x3).
# 3-stage softKD replay. Total ~9.5-9.85 H100h incl. teacher inference (< 10 cap).
# All pools are in-repo paths on artemis NFS; all are decontaminated + position-balanced.
set -euo pipefail
REPO=/data/nfs/ruggsea/italic-challenge

# ---- Stage 1: committee softKD from base -> 0.4787 (~4.7 H100h)
# (identical to code/final_run.sh — the compression recipe on grammar_committee_fullrich_v2)
bash $REPO/submission/code/final_run.sh

# ---- Stage 2 (M2): culture softKD from stage-1 -> 0.4878/0.4880 (~2.0-2.2 H100h)
python3 $REPO/scripts/train_softkd.py \
  --student <STAGE1_BEST_CKPT> \
  --data $REPO/data/pools/grammar/m2_committee_softkd_t61.jsonl \
  --temp 2.0 --lmbda-ce 0.5 --lr 2e-5 --seed 99 --max-steps 2000 \
  --out $REPO/experiments/m2-replay
# best ckpt = step 2000 (dev2k-official 0.499)

# ---- Stage 3 (extkd): external in-distribution softKD from M2-s2000 -> 0.492-0.493 (~2.5 H100h)
# Pool: external_combined_t94 committee-labeled (28,561 items w/ teacher_logprobs; labeling
# = Mistral-Small-3.2-24B local serve, ~0.15 H100h — see scripts/hel_committee_label_m2_t58.sh
# pattern; overlap vs test verified 0% exact / 0% cosine>=0.80)
python3 $REPO/scripts/train_softkd.py \
  --student $REPO/experiments/m2-replay/ckpt_step2000 \
  --data $REPO/data/pools/external/extkd_committee_softkd_t98.jsonl \
  --temp 2.0 --lmbda-ce 0.5 --lr 2e-5 --seed 99 --max-steps 1500 \
  --out $REPO/experiments/extkd-replay
# best ckpt = step 1500 (dev2k-official 0.501)

# ---- Eval (official, unmodified harness)
# python3 $REPO/eval/run_eval.py --model <CKPT> --data $REPO/eval/italic.jsonl --fast --max-tokens 350 --temp 0
