#!/bin/bash
# Serve Mistral-Small-3.2-24B once on one H200 and SYNTHESIZE a large diverse batch of
# Italian ORTHOGRAPHY + MORPHOLOGY MCQs (the starved grammar floor). Then re-index into an
# index-aligned candidate file for committee labeling. Resumable (gen counts existing lines).
#   sbatch gsc1_orthomorph_gen.sbatch
#SBATCH --job-name=om-gen
#SBATCH -p idealab_cs2
#SBATCH -A idealab_cs2
#SBATCH --gres=gpu:h200.140gb:1
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH -t 08:00:00
#SBATCH --output=/cl_tmp/lazzaron/italic-star/logs/om-gen-%j.out
set -uo pipefail
export PATH=/opt/slurm/rocky-9.6/*/bin:$PATH
export HF_HOME=/cl_tmp/lazzaron/hf-cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR=/cl_tmp/lazzaron/tmp
mkdir -p "$TMPDIR"

ROOT=/cl_tmp/lazzaron/italic-star
VENV=/cl_tmp/lazzaron/vllm-venv
EVAL=/cl_tmp/lazzaron/italic-eval-venv
ITALIC=/cl_tmp/lazzaron/ITALIC
MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
NTFY=ruggsea-claude-gsc1-guide
N_ORTHO=${N_ORTHO:-3000}
N_MORPH=${N_MORPH:-3000}
WORK=$ROOT/data/pools/grammar
SYNTH=$WORK/orthomorph_synth.jsonl
CAND=$WORK/orthomorph_candidates.jsonl
mkdir -p "$WORK"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "[om-gen] MODEL=$MODEL PORT=$PORT node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES n_ortho=$N_ORTHO n_morph=$N_MORPH"

$VENV/bin/vllm serve "$MODEL" --served-model-name "$MODEL" \
  --tensor-parallel-size 1 --port $PORT \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
  > "$WORK/om_serve.log" 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 180); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "[om-gen] serve died"; tail -50 "$WORK/om_serve.log"; exit 1; }
  sleep 10
done
[ $ok = 1 ] || { echo "[om-gen] health timeout"; tail -50 "$WORK/om_serve.log"; kill $SPID; exit 1; }
echo "[om-gen] server healthy"

$EVAL/bin/python "$ROOT/scripts/gen_ortho_morph.py" \
  --n-ortho $N_ORTHO --n-morph $N_MORPH --out "$SYNTH" \
  --model "$MODEL" --base-url http://localhost:$PORT/v1 --threads 64 \
  || { echo "FATAL gen rc=$?"; kill $SPID; exit 2; }
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null

# re-index 0..N-1 into an index-aligned candidate file (label_committee.py uses 'index')
$EVAL/bin/python - "$SYNTH" "$CAND" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src) if l.strip()]
for i, r in enumerate(rows):
    r["index"] = i
with open(dst + ".tmp", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
import os; os.replace(dst + ".tmp", dst)
import collections
print(f"[cand] wrote {len(rows)} -> {dst}  cats={dict(collections.Counter(r['category'] for r in rows))}")
PY
NCAND=$(wc -l < "$CAND")
echo "[om-gen] DONE candidates=$NCAND -> $CAND"
curl -s -H "Email: rmlazza@gmail.com" -H "Title: ITALIC ortho/morph synth done" \
  -d "ortho/morph synth done: candidates=$NCAND at $CAND. Committee-labeling chained." ntfy.sh/$NTFY || true
