#!/usr/bin/env python
"""Label an MCQ pool with a teacher served over an OpenAI-compatible endpoint.

Prompts are byte-identical to the ITALIC 5-shot-fast eval: we import the
harness's own `configure_payload` (from eval/run_eval.py) with the same
5_shots.jsonl few-shot block, fast letter-only template, temp 0. For each item
we capture the teacher's greedy first token (its answer letter) AND the top-k
first-token logprobs over option letters. That single artifact unblocks both
distillation routes: teacher-correct hard-label filtering (letter == gold) and
soft-label KL (the letter logprob distribution).

Idempotent/resumable: appends one JSON line per item to --out; on restart it
skips indices already present. Fails loudly (no retry-hiding beyond transient
network via tenacity, matching the harness).

Usage (run inside the eval venv, teacher already served):
  python label_teacher.py --pool POOL.jsonl --few-shot 5_shots.jsonl \
    --out labels.jsonl --model MODEL --base-url http://localhost:PORT/v1 \
    [--eval-dir DIR_WITH_run_eval.py] [--threads 64] [--top-logprobs 20] \
    [--system "..."] [--limit N]
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import tenacity
from openai import OpenAI
from tqdm import tqdm


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def done_indices(path):
    if not os.path.exists(path):
        return set()
    seen = set()
    for l in open(path):
        seen.add(json.loads(l)["index"])
    return seen


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=1, max=8),
    stop=tenacity.stop_after_attempt(20),
)
def query(client, model, messages, top_logprobs):
    # max_tokens=1: the answer is the single letter; temp 0 => greedy = argmax.
    r = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=1,
        logprobs=True, top_logprobs=top_logprobs,
    )
    choice = r.choices[0]
    tok = choice.logprobs.content[0]
    dist = {t.token: t.logprob for t in tok.top_logprobs}
    return choice.message.content, tok.token, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--few-shot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--eval-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/eval",
                    help="dir containing run_eval.py (for configure_payload)")
    ap.add_argument("--system", default=None, help="teacher system message; default 'Sei un assistente utile.'")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.eval_dir)
    from run_eval import configure_payload, extract_answer_fast  # byte-identical prompts

    pool = load_jsonl(args.pool)
    if args.limit:
        pool = pool[: args.limit]
    few_shots = load_jsonl(args.few_shot)
    seen = done_indices(args.out)
    todo = [(i, it) for i, it in enumerate(pool) if i not in seen]
    print(f"[label] pool={len(pool)} done={len(seen)} todo={len(todo)} model={args.model}", flush=True)
    if not todo:
        print("[label] nothing to do", flush=True)
        return

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    lock = threading.Lock()
    out = open(args.out, "a")

    def work(idx_item):
        i, it = idx_item
        sample = configure_payload(
            topic=it["category"], question=it["question"], options=it["options"],
            answer=it["answer"], fast=True, system_message=args.system, few_shots=few_shots,
        )
        raw, first_tok, dist = query(client, args.model, sample.messages, args.top_logprobs)
        letter = extract_answer_fast(raw)  # official first-letter extractor
        rec = {"index": i, "category": it["category"], "gold": it["answer"],
               "teacher_letter": letter, "correct": letter == it["answer"],
               "first_token": first_tok, "top_logprobs": dist}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    correct = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(work, x) for x in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc="labeling"):
            f.result()  # re-raise any error loudly

    out.close()
    # report accuracy over everything labeled so far
    tot = cor = 0
    for l in open(args.out):
        r = json.loads(l); tot += 1; cor += int(r["correct"])
    print(f"[label] done. labeled={tot} teacher_correct={cor} acc={cor/tot:.4f}", flush=True)


if __name__ == "__main__":
    main()
