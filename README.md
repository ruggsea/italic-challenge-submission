# ITALIC Post-Training Challenge submission

Submission for the [mii-llm Italian Post-Training Challenge](https://huggingface.co/spaces/mii-llm/Post-Training-Challenge): post-train [mii-llm/zagreus-0.4B-ita](https://huggingface.co/mii-llm/zagreus-0.4B-ita) (437M parameters) to maximize accuracy on [ITALIC](https://github.com/Crisp-Unimib/ITALIC), a 10,000-question Italian culture and language benchmark, with a final-run budget of 10 H100 GPU hours.

**Result: 0.4787 accuracy on the full 10K** (official unmodified harness, 5-shot fast mode, temperature 0). Confirmed on three independent evaluation runs: 0.4787, 0.4784, and a third verification run. The base model scores 0.2802; the public leaderboard best at submission time was 0.372.

**Model:** [idealab-cs2/zagreus-0.4B-italic-softkd](https://huggingface.co/idealab-cs2/zagreus-0.4B-italic-softkd)

## Method

The base model answers ITALIC questions at roughly the gold-A/B marginal (~0.28): it emits almost only the letters A and B and carries very little discriminative signal. Standard approaches all failed here, in instructive ways (see [REPORT.md](REPORT.md) for the full negative-results list): hard-label SFT collapses the model to a constant letter, continued pretraining does not make answers decodable at this scale, and sequence-level distillation from a stronger model destroys what little signal exists.

What works is **soft answer-distribution knowledge distillation (softKD)**: for every training question, a teacher provides its probability distribution over just the option letters (A..J) at the answer position, and the student is trained with a KL term on that single token (temperature 2.0) plus cross-entropy on the gold letter (weight 0.5), in fp32. Because only the letter distribution is transferred, this is robust to teacher/student tokenizer and lineage mismatch.

The final recipe is one training run from the base model:

1. **Teacher committee.** Three teachers, each fitting a single H100 80GB in bf16: Mistral-Small-3.2-24B, Gemma-3-27B, Qwen3-32B. Their averaged letter distributions score 0.77 on the training pool, better than any single teacher.
2. **Training pool** ([ruggsea/italic-softkd-pool](https://huggingface.co/datasets/idealab-cs2/italic-softkd-pool), 21,606 items, committee soft labels included — with provenance splits knowledge/grammar/orthomorph as named splits): 11,453 knowledge MCQs (pinocchio-derived, teacher-filtered), 5,432 committee-labeled grammar MCQs, and ~4,700 generated orthography/morphology MCQs targeting the categories where public data is scarce. Everything is decontaminated against the ITALIC test set (exact match plus semantic similarity at cosine 0.80; see `data/decontam_manifest.json`, 0 exact leaks, max similarity 0.7999).
3. **One softKD run** from base (chatml chat template embedded, LR 1.5e-4, up to 4000 steps, best checkpoint by a fixed 2K-question dev subset, early stopping).
4. **Official evaluation** on the full 10K with a fresh vLLM server.

Two details matter more than they look. The chat template is worth about +4 points and is legal surface (it ships with the model, the evaluation code is untouched). And the student must not train in pure bf16 at small learning rates: AdamW updates land below the bf16 representable step and the weights never move. All our early distillation failures trace back to this.

## Reproducing the final run (≈4.7 H100 hours)

`code/final_run.sh` is the pipeline: build base+template, one softKD training job, official eval. Measured costs on a single H200 (converted at H200 = 1.2 H100):

| Stage | Wall clock (1x H200) | H100h |
|---|---|---|
| Grammar MCQ generation, Mistral-24B (`gen_grammar_mistral.py`, 4,000 items + first-pass labels) | 5 min | 0.10 |
| Ortho/morph MCQ generation, Mistral-24B (`gen_ortho_morph.py`, 6,000 items) | 7 min | 0.15 |
| WordNet / traditional-NLP grammar candidates (CPU) | - | 0 |
| 3-teacher committee labeling: grammar candidates (11,793) | 8 min | 0.16 |
| 3-teacher committee labeling: ortho/morph candidates (6,000) | 7 min | 0.13 |
| 3-teacher committee labeling: knowledge subset (11,453; measured-equivalent job) | ~9 min | 0.18 |
| softKD training run (`final_train.slurm`) | 3h05m | 3.70 |
| Official full-10K eval (`gsc1_eval_only.slurm`) | 14 min | 0.28 |
| Total | | **≈4.8** |

The soft-label pool is on Hugging Face (italic-softkd-pool), so a replay can also skip labeling and still stay within budget with room to spare. Evaluation uses the official `run_eval.py` from the ITALIC repo, unmodified, fast mode, 5-shot, temperature 0.

## Repository layout

- `code/` training, data-generation, labeling and eval scripts
- `data/` the decontamination manifest; the training data itself is on Hugging Face (below)
- `assets/chat-template/` the chat template and tokenizer config embedded in the model
- `eval_results/` official harness outputs for the shipped model and confirmations
- `REPORT.md` full report: idea, experiments, what worked, what did not

## Rules compliance

- Started from `mii-llm/zagreus-0.4B-ita`, no external checkpoints used as initialization.
- Evaluation code unmodified (clean clone of Crisp-Unimib/ITALIC).
- No training on ITALIC test items: exact and semantic decontamination of every pool, manifest included.
- All teachers fit one H100 80GB in bf16; all synthetic data generated locally on GPU, no API calls.
- Final run replays in ≈4.7 H100 hours, under the 10-hour budget.

## Acknowledgments

Done by [ruggsea](https://huggingface.co/ruggsea), using compute resources of the Complex Social & Computational Systems (CS²) group, IDea_Lab, University of Graz.

## License

Apache 2.0, same as the base model.
