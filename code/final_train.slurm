#!/bin/bash
#SBATCH --job-name=final-clean
#SBATCH --partition=idealab_cs2
#SBATCH --account=idealab_cs2
#SBATCH --gres=gpu:h200.140gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/cl_tmp/lazzaron/italic-star/logs/final-%j.out
# Compression-fallback FINAL CLEAN RUN: base(+chatml)/seed -> softKD(committee soft labels) in ONE run.
#   sbatch --export=ALL,TAG=..,STUDENT=..,DATA=..,LR=..,MAXSTEPS=.. final_train.slurm
set -uo pipefail
export PATH=/opt/slurm/rocky-9.6/*/bin:$PATH
export HF_HOME=/cl_tmp/lazzaron/hf-cache
export TMPDIR=/cl_tmp/lazzaron/tmp; mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false
RUN=/cl_tmp/lazzaron/italic-star; cd "$RUN"
PY=/cl_tmp/lazzaron/vllm-venv/bin/python
: "${TAG:?}"; : "${STUDENT:?}"; : "${DATA:?}"; : "${LR:=1.5e-4}"; : "${MAXSTEPS:=4000}"
echo "[final] TAG=$TAG STUDENT=$STUDENT DATA=$DATA LR=$LR MAXSTEPS=$MAXSTEPS start=$(date -Is)"
$PY scripts/train_softkd.py --data "$DATA" --tag "$TAG" --student "$STUDENT" \
  --shots eval/5_shots.jsonl --dev-file data/italic_dev2k.jsonl \
  --temp 2.0 --lmbda-ce 0.5 --lr "$LR" --bs 8 --grad-accum 4 \
  --max-steps "$MAXSTEPS" --eval-steps 150 --patience 10
echo "[final] train done=$(date -Is) exit=$?"
ls -la experiments/$TAG/model/ 2>/dev/null
