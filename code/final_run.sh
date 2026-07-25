#!/bin/bash
# FINAL CLEAN RUN — compression-fallback reproducible pipeline (per user directive 2026-07-25).
#
# The faithful winning chain (base -> committee_rich round -> iterate-to-convergence 0.4387
# seed -> additive ortho/morph superset 0.4652) is ~5-6 distillation rounds and would exceed
# the 10 H100h replay budget. So we use the mandated COMPRESSION FALLBACK: distill the FINAL
# committee soft labels (already embedded in grammar_committee_fullrich_v2.jsonl) directly from
# base in ONE softKD training run, then score the full official 10K. If from-base reproduces
# >=0.45 it is the minimal recipe; a from-r2union (from-base content-SFT seed) 2-stage variant
# is launched in parallel as a safer fallback (both << 10 H100h).
#
# Data provenance (all decontaminated: 0 exact, 0 semantic cos>=0.80 vs eval/italic.jsonl):
#   grammar_committee_fullrich_v2.jsonl (21606 items, committee soft labels = 3-teacher ensemble
#   mistral-24b + gemma-27b + qwen3-32b, each fits 1x H100 80GB): committee_rich knowledge (11453)
#   + committee-labeled grammar (5432) + generated+committee-labeled ortho/morph MCQs (~4721,
#   attacking the starved orthography(92)/morphology(563) cats). Teacher labeling ~30 min on 1 H200.
#
# Template enforcement (directive 1): base weights get the leader-style chatml template embedded
# (base-chatml/model = base safetensors + chat_template.jinja/tokenizer from a trained seed);
# every checkpoint embeds it, eval serves with the embedded template.
#
# Run on GSC1 idealab_cs2 (1x H200-143GB). H200->H100 factor 1.2 (log in BUDGET.md).
set -euo pipefail
# Site-specific: point RUN at your working dir and SNAP at the local HF snapshot of mii-llm/zagreus-0.4B-ita
RUN=/cl_tmp/lazzaron/italic-star
SNAP=/cl_tmp/lazzaron/hf-cache/hub/models--mii-llm--zagreus-0.4B-ita/snapshots/f463a5beabf054809ee3cc702c711215a5fde690
ASSETS=$(cd "$(dirname "$0")/../assets/chat-template" && pwd)  # chatml template + tokenizer shipped with this repo
DATA=data/pools/grammar_committee_fullrich_v2.jsonl
cd "$RUN"; export PATH=/opt/slurm/rocky-9.6/*/bin:$PATH

# --- Stage 0: build base + embedded chatml template ---
DST=experiments/base-chatml/model
if [ ! -f "$DST/model.safetensors" ]; then
  mkdir -p "$DST"
  cp "$SNAP"/config.json "$SNAP"/generation_config.json "$SNAP"/model.safetensors "$DST"/
  cp "$ASSETS"/chat_template.jinja "$ASSETS"/tokenizer_config.json "$ASSETS"/tokenizer.json "$DST"/
fi

# --- Stage 1: ONE softKD run from base (compression) + full official 10K eval, armed afterok ---
JA=$(sbatch --parsable --export=ALL,TAG=final-clean-frombase,STUDENT=experiments/base-chatml/model,DATA=$DATA,LR=1.5e-4,MAXSTEPS=4000 final_train.slurm)
sbatch --dependency=afterok:$JA --export=ALL,TAG=final-clean-frombase-full,MODEL=$RUN/experiments/final-clean-frombase/model,DATA=$RUN/eval/italic.jsonl gsc1_eval_only.slurm

# --- Fallback variant: from-base content-SFT (r2union) seed -> softKD (2-stage, safer start) ---
JB=$(sbatch --parsable --export=ALL,TAG=final-clean-r2union,STUDENT=experiments/undertrain-r2union/model,DATA=$DATA,LR=1.5e-4,MAXSTEPS=3000 final_train.slurm)
sbatch --dependency=afterok:$JB --export=ALL,TAG=final-clean-r2union-full,MODEL=$RUN/experiments/final-clean-r2union/model,DATA=$RUN/eval/italic.jsonl gsc1_eval_only.slurm

echo "final-clean launched: frombase train=$JA / r2union train=$JB (+armed full-10K evals). Harvest score.json when done."
# The variant reproducing >=0.45 at the lowest budget is THE submission artifact.
