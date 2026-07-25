#!/bin/bash
# Committee-label the new ortho/morph candidate set with the 3 finalists
# (mistral-24b + gemma-27b + qwen3-32b) SEQUENTIALLY on ONE H200 (TP=1). Idempotent:
# a teacher whose output jsonl exists is skipped. Chained afterok the gen job.
#SBATCH -p idealab_cs2
#SBATCH -A idealab_cs2
#SBATCH --gres=gpu:h200.140gb:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 05:00:00
#SBATCH -J om-label
#SBATCH --output=/cl_tmp/lazzaron/italic-star/logs/om-label-%j.out
set -uo pipefail
export PATH=/opt/slurm/rocky-9.6/*/bin:$PATH
export HF_HOME=/cl_tmp/lazzaron/hf-cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export TMPDIR=/cl_tmp/lazzaron/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$TMPDIR"

ROOT=/cl_tmp/lazzaron/italic-star
export ITALIC_REPO=$ROOT
PY=/cl_tmp/lazzaron/vllm-venv/bin/python
LABEL=$ROOT/tracks/committee/scripts/label_committee.py
DATA=$ROOT/data/pools/grammar/orthomorph_candidates.jsonl
OUTDIR=$ROOT/tracks/committee/labels/orthomorph
mkdir -p "$OUTDIR"
echo "[om-label] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES data=$DATA n=$(wc -l < $DATA)"

run_one() {  # KEY HF_ID EXTRA...
  local key="$1" model="$2"; shift 2
  local out="$OUTDIR/$key.jsonl"
  if [ -f "$out" ]; then echo "[skip] $key exists"; return 0; fi
  echo "[run] $key extra=[$*]"
  $PY "$LABEL" --model "$model" --data "$DATA" --out "$out" \
    --tp 1 --gpu-mem 0.90 --max-model-len 4096 "$@" > "$OUTDIR/$key.log" 2>&1
  echo "[done] $key rc=$?"
}

run_one mistral-24b mistralai/Mistral-Small-3.2-24B-Instruct-2506 --mistral-fmt
run_one gemma-27b   google/gemma-3-27b-it
run_one qwen3-32b   Qwen/Qwen3-32B --no-think --sysmsg "Sei un assistente utile. /no_think"
echo "[om-label] ALL DONE"; wc -l "$OUTDIR"/*.jsonl
curl -s -H "Email: rmlazza@gmail.com" -H "Title: ITALIC: ortho/morph committee-labeling done" \
  -d "ortho/morph candidates labeled (mistral+gemma+qwen3) in $OUTDIR" ntfy.sh/ruggsea-claude-gsc1-guide >/dev/null 2>&1 || true
