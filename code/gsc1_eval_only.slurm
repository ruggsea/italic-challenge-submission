#!/bin/bash
#SBATCH --job-name=italic-eval
#SBATCH --partition=idealab_cs2
#SBATCH --account=idealab_cs2
#SBATCH --gres=gpu:h200.140gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/cl_tmp/lazzaron/italic-star/logs/eval-%j.out
# Eval-only on GSC1 (byte-exact stack for transformers-5.14.1 checkpoints that
# the artemis vLLM 0.x tokenizer loader cannot parse). Serves an already-trained
# GSC1 checkpoint with its EMBEDDED chat template and scores it with the UNMODIFIED
# official eval/run_eval.py. Writes score.json with accuracy + pred-distribution +
# per-gold-letter + per-category (collapse guard).
#   sbatch --export=ALL,TAG=<tag>,MODEL=<dir>,DATA=<jsonl> gsc1_eval_only.slurm
# Idempotent: skips if $SCORE already exists.
set -uo pipefail
export PATH=/opt/slurm/rocky-9.6/*/bin:$PATH:/cl_tmp/lazzaron/vllm-venv/bin
export HF_HOME=/cl_tmp/lazzaron/hf-cache
export TMPDIR=/cl_tmp/lazzaron/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"

RUN=/cl_tmp/lazzaron/italic-star
VENV=/cl_tmp/lazzaron/vllm-venv
EVAL=/cl_tmp/lazzaron/italic-eval-venv
NTFY=ruggsea-claude-gsc1-guide
: "${TAG:?need TAG}"; : "${MODEL:?need MODEL}"; : "${DATA:?need DATA (jsonl)}"
cd "$RUN"; mkdir -p logs
OUT=$RUN/experiments/$TAG
SCORE=$OUT/score.json
mkdir -p "$OUT"
[ -f "$SCORE" ] && { echo "[eval] SKIP: $SCORE exists"; exit 0; }

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "[eval] tag=$TAG model=$MODEL data=$DATA port=$PORT node=$(hostname)"
$VENV/bin/vllm serve "$MODEL" --served-model-name italic_model \
  --tensor-parallel-size 1 --port $PORT --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager > logs/$TAG.serve.log 2>&1 &
SPID=$!
ok=0
for i in $(seq 1 180); do
  curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "[eval] serve died"; tail -40 logs/$TAG.serve.log; exit 6; }
  sleep 10
done
[ $ok = 1 ] || { echo "[eval] serve health timeout"; tail -40 logs/$TAG.serve.log; kill $SPID; exit 6; }

RESULTS=$OUT/results; mkdir -p "$RESULTS"
CFG=$OUT/config.yaml
cat > "$CFG" <<YAML
provider: custom_openai
model: italic_model
api_key: EMPTY
temperature: 0.0
max_tokens: 350
system_message: null
provider_kwargs:
  base_url: http://localhost:$PORT/v1
rate_limiting:
  enabled: false
  requests_per_minute: 15
limit: null
fast: true
data:
  data_file: $DATA
  few_shot_file: $RUN/eval/5_shots.jsonl
  output_dir: $RESULTS
num_threads: 64
auto_resume: true
checkpointing:
  enabled: true
  checkpoint_interval: 500
YAML

START=$(date +%s)
$EVAL/bin/python eval/run_eval.py --config-path "$OUT" --config-name config \
  hydra.run.dir="$OUT/hydra" hydra.output_subdir=null hydra.job.chdir=False
RC=$?
ELAPSED=$(( $(date +%s) - START ))
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
[ "$RC" = 0 ] || { echo "FATAL eval rc=$RC"; tail -40 logs/$TAG.serve.log; exit 7; }

$EVAL/bin/python - "$RESULTS" "$OUT" "$MODEL" "$DATA" "$ELAPSED" "$RUN/eval" <<'PY'
import glob, json, os, sys, datetime, collections
results_dir, exp_dir, model, data_file, elapsed, evaldir = sys.argv[1:7]
sys.path.insert(0, evaldir)
from run_eval import extract_answer_fast
metrics=None; src=None; results=None
for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if p.endswith("_checkpoint.json"): continue
    d=json.load(open(p))
    if isinstance(d, dict) and "metrics" in d:
        metrics=d["metrics"]; src=p; results=d.get("results", [])
if metrics is None:
    print("FATAL: no metrics file", results_dir, file=sys.stderr); sys.exit(5)
n=len(results)
cats=[json.loads(l)["category"] for l in open(data_file)]
per=collections.defaultdict(lambda:[0,0])
pred_dist=collections.Counter(); per_gold=collections.defaultdict(lambda:[0,0])
for r in results:
    pred=extract_answer_fast(r["output"]) if r.get("fast") else r["output"]
    cat=cats[r["index"]] if r["index"]<len(cats) else "?"
    per[cat][1]+=1; per[cat][0]+=int(pred==r["answer"])
    pred_dist[pred or "<none>"]+=1
    g=r["answer"]; per_gold[g][1]+=1; per_gold[g][0]+=int(pred==g)
per_cat={k:{"acc":round(v[0]/v[1],4),"n":v[1]} for k,v in sorted(per.items())}
per_gold_acc={g:{"acc":round(c/t,4),"n":t} for g,(c,t) in sorted(per_gold.items())}
summary={"tag":os.path.basename(exp_dir),"model":model,"data_file":data_file,
 "n_examples":n,"accuracy":metrics.get("accuracy"),"accuracy_std":metrics.get("accuracy:std"),
 "prediction_distribution":dict(sorted(pred_dist.items())),"per_gold_letter":per_gold_acc,
 "per_category":per_cat,"results_json":src,"eval_wallclock_seconds":int(float(elapsed)),
 "timestamp":datetime.datetime.now().isoformat(timespec="seconds")}
tmp=os.path.join(exp_dir,"score.json.tmp")
json.dump(summary, open(tmp,"w"), indent=2, ensure_ascii=False)
os.replace(tmp, os.path.join(exp_dir,"score.json"))
print(f"[eval] ACCURACY={summary['accuracy']:.4f} n={n} preds={summary['prediction_distribution']}")
print(f"[eval] per-gold {per_gold_acc}")
PY
ACC=$(python3 -c "import json;print(json.load(open('$SCORE'))['accuracy'])" 2>/dev/null)
curl -s -H "Email: rmlazza@gmail.com" -H "Title: ITALIC eval: $TAG=$ACC" \
  -d "$TAG on $(basename $DATA): acc=$ACC. See score.json for pred-dist + per-gold." ntfy.sh/$NTFY || true
echo "[eval] done: $SCORE"
