#!/usr/bin/env python
"""Synthesize a LARGE, diverse batch of Italian ORTHOGRAPHY + MORPHOLOGY MCQs.

These two grammar categories are the ITALIC floor AND the starved ones: after
committee-labeling only 92 orthography + 563 morphology items survive teacher-correct.
The additive-grammar arms cannot fix a category with no data -> generate much more.

Design vs gen_grammar_mistral.py:
  * ONLY orthography + morphology, each split into many fine-grained real phenomena
    (accents, doubles, apostrophe/elision, h, sce/scie, cqu, -cia/-gia plurals,
    irregular plurals/feminines, superlatives/comparatives, verb mood+tense,
    irregular participles/passato-remoto, prefix/root/suffix).
  * The specific tested word/phrase is embedded IN THE QUESTION STEM so every item
    has a distinct stem -> survives the question-only dedup in the downstream
    committee-pool builder (build_committee_softkd_pool.py dedups by question text).
  * Big word-seed bank + high temp for lexical diversity.
  * PER-CATEGORY targets: generate until BOTH orthography and morphology floors hit.
  * dedup by normalized question+options (robust); teacher-correctness enforced
    downstream by the 3-teacher committee re-labeling at temp 0.

Idempotent/resumable: appends accepted items to --out; on restart counts existing
per-category lines and generates the remainder. Fails loudly.

Usage (inside eval venv, teacher served OpenAI-compatible):
  python gen_ortho_morph.py --n-ortho 3000 --n-morph 3000 --out orthomorph_synth.jsonl \
    --model MODEL --base-url http://localhost:PORT/v1 --threads 64
"""
import argparse, json, os, random, re, threading, unicodedata, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import tenacity
from openai import OpenAI
from tqdm import tqdm

# subtype -> (category, share_within_category, instruction, few-shot exemplars).
# Exemplars are hand-written and NOT ITALIC items (decontam re-verifies downstream).
SUBTYPES = {
    # ---------------- ORTHOGRAPHY ----------------
    "ortho_accento": ("orthography", 0.16,
        "Genera UNA domanda a scelta multipla di ortografia italiana sull'ACCENTO: dato un "
        "significato o un contesto, si chiede quale sia la grafia CORRETTA di una parola che "
        "puo' richiedere l'accento (es. perche', poiche', affinche', citta', virtu', pero', se', "
        "ne', da'/da, e'/e, si'/si, la'/la). Metti la parola o il suo significato NELLA domanda "
        "cosi' che sia autoconsistente. Una sola opzione corretta, le altre con accento sbagliato o assente.",
        [{"question": 'Quale grafia e\' corretta per la congiunzione causale che significa "dato che"?',
          "options": {"A": "perche'", "B": "perche", "C": "perchè", "D": "per che'"}, "answer": "A"}]),
    "ortho_apostrofo": ("orthography", 0.16,
        "Genera UNA domanda a scelta multipla di ortografia italiana su APOSTROFO ed ELISIONE/TRONCAMENTO: "
        "dato un breve contesto o sintagma, si chiede quale sia la forma CORRETTA (es. un'amica vs un amica, "
        "un po' vs un po vs un po', qual e' vs qual'e', d'accordo, dev'essere, po', be'). Inserisci il "
        "sintagma NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Come si scrive correttamente il sintagma con "poco" nella frase "Aspetta ..."?',
          "options": {"A": "un po'", "B": "un po", "C": "un pò", "D": "un’po"}, "answer": "A"}]),
    "ortho_doppie": ("orthography", 0.13,
        "Genera UNA domanda a scelta multipla di ortografia italiana sulle CONSONANTI DOPPIE: dato il "
        "significato di una parola, si chiede quale sia la grafia corretta tra varianti con doppie giuste o "
        "sbagliate (es. avvenimento, commozione, appartamento, obbligo, soqquadro, ebbene). Metti il "
        "significato o un sinonimo NELLA domanda. Una sola corretta.",
        [{"question": 'Quale grafia e\' corretta per il sostantivo che indica un forte turbamento emotivo?',
          "options": {"A": "commozione", "B": "comozione", "C": "commossione", "D": "comossione"}, "answer": "A"}]),
    "ortho_h": ("orthography", 0.10,
        "Genera UNA domanda a scelta multipla di ortografia italiana sull'uso dell'ACCA (h): dato un contesto, "
        "si chiede la forma corretta tra coppie omofone (ho/o, hai/ai, ha/a, hanno/anno, ahi/ai, oh/o). "
        "Inserisci la frase NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Nella frase "... comprato il pane" (io, tempo presente), quale forma verbale e\' corretta?',
          "options": {"A": "Ho", "B": "O", "C": "Oh", "D": "Hò"}, "answer": "A"}]),
    "ortho_gruppi": ("orthography", 0.13,
        "Genera UNA domanda a scelta multipla di ortografia italiana sui GRUPPI ORTOGRAFICI difficili "
        "(sce/scie, ce/cie, ge/gie, cqu/qqu, gn/gni, gli, mb/mp): dato il significato, si chiede la grafia "
        "corretta (es. scienza, coscienza, efficiente, sufficiente, acqua, soqquadro, ingegnere). Metti il "
        "significato NELLA domanda. Una sola corretta.",
        [{"question": 'Quale grafia e\' corretta per il termine che indica la consapevolezza morale di se\'?',
          "options": {"A": "coscienza", "B": "coscenza", "C": "cosciensa", "D": "conscienza"}, "answer": "A"}]),
    "ortho_plurale_grafia": ("orthography", 0.10,
        "Genera UNA domanda a scelta multipla di ortografia italiana sul PLURALE di parole in -cia/-gia o "
        "-io: dato il singolare, si chiede la grafia corretta del plurale (camicia->camicie, ciliegia->ciliegie, "
        "spiaggia->spiagge, valigia->valigie/valige, farmacia->farmacie, bacio->baci, zio->zii). Metti il "
        "singolare NELLA domanda. Una sola corretta.",
        [{"question": 'Qual e\' la grafia corretta del plurale di "camicia"?',
          "options": {"A": "camicie", "B": "camice", "C": "camicee", "D": "camici"}, "answer": "A"}]),
    "ortho_grafia_parola": ("orthography", 0.12,
        "Genera UNA domanda a scelta multipla di ortografia italiana in cui, dato il SIGNIFICATO di una parola "
        "insidiosa, si chiede quale delle quattro varianti sia scritta correttamente (una sola). Varia molto le "
        "parole (termini colti, tecnici, quotidiani). Metti il significato NELLA domanda per renderla univoca.",
        [{"question": 'Quale delle seguenti e\' la grafia corretta della parola che indica chi ripara le scarpe?',
          "options": {"A": "calzolaio", "B": "calzolaro", "C": "calzolajo", "D": "calzoraio"}, "answer": "A"}]),
    # ---------------- MORPHOLOGY ----------------
    "morph_plurale_irreg": ("morphology", 0.14,
        "Genera UNA domanda a scelta multipla di morfologia italiana sul PLURALE IRREGOLARE o sovrabbondante: "
        "dato il singolare, si chiede il plurale corretto (uomo->uomini, dio->dei, bue->buoi, ala->ali, "
        "arma->armi, uovo->uova, dito->dita, il braccio->le braccia, tempio->templi, uomo). Metti il singolare "
        "NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Qual e\' il plurale corretto del sostantivo "uovo"?',
          "options": {"A": "uova", "B": "uovi", "C": "uove", "D": "ove"}, "answer": "A"}]),
    "morph_femminile": ("morphology", 0.11,
        "Genera UNA domanda a scelta multipla di morfologia italiana sul FEMMINILE irregolare di un nome: "
        "dato il maschile, si chiede il femminile corretto (attore->attrice, eroe->eroina, gallo->gallina, "
        "poeta->poetessa, dio->dea, re->regina, cane->cagna, dottore->dottoressa). Metti il maschile NELLA "
        "domanda. Una sola corretta.",
        [{"question": 'Qual e\' il femminile corretto del nome "eroe"?',
          "options": {"A": "eroina", "B": "eroessa", "C": "eroa", "D": "eroina’"}, "answer": "A"}]),
    "morph_superlativo": ("morphology", 0.14,
        "Genera UNA domanda a scelta multipla di morfologia italiana sul SUPERLATIVO ASSOLUTO (spesso "
        "irregolare/dotto) di un aggettivo: dato l'aggettivo, si chiede il superlativo corretto "
        "(celebre->celeberrimo, acre->acerrimo, misero->miserrimo, salubre->saluberrimo, integro->integerrimo, "
        "buono->ottimo/buonissimo, grande->massimo/grandissimo). Metti l'aggettivo NELLA domanda. Una sola corretta.",
        [{"question": 'Qual e\' il superlativo assoluto corretto dell\'aggettivo "celebre"?',
          "options": {"A": "celeberrimo", "B": "celebrissimo", "C": "celeberrimo", "D": "celeberissimo"}, "answer": "A"}]),
    "morph_comparativo": ("morphology", 0.09,
        "Genera UNA domanda a scelta multipla di morfologia italiana sul COMPARATIVO/superlativo organico "
        "(irregolare) di aggettivi come buono, cattivo, grande, piccolo, alto, basso (migliore, peggiore, "
        "maggiore, minore, superiore, inferiore, ottimo, pessimo). Metti l'aggettivo e il grado richiesto "
        "NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Qual e\' il comparativo di maggioranza organico dell\'aggettivo "buono"?',
          "options": {"A": "migliore", "B": "più buono", "C": "buonissimo", "D": "meglio"}, "answer": "A"}]),
    "morph_verbo_modo_tempo": ("morphology", 0.16,
        "Genera UNA domanda a scelta multipla di morfologia verbale italiana in cui, data una VOCE VERBALE "
        "coniugata, si chiede a quale MODO e TEMPO appartiene (es. 'avremmo parlato' = condizionale passato; "
        "'che io fossi' = congiuntivo imperfetto; 'ebbe fatto' = trapassato remoto; 'avrai visto' = futuro "
        "anteriore). Varia molto i verbi. Metti la voce verbale NELLA domanda. Una sola opzione corretta.",
        [{"question": 'La voce verbale "avremmo parlato" e\' al modo e tempo:',
          "options": {"A": "condizionale passato", "B": "congiuntivo trapassato", "C": "indicativo futuro anteriore", "D": "condizionale presente"}, "answer": "A"}]),
    "morph_coniugazione": ("morphology", 0.11,
        "Genera UNA domanda a scelta multipla di morfologia verbale italiana in cui si chiede la FORMA "
        "CORRETTA di un verbo coniugato a un dato modo/tempo/persona, prediligendo verbi irregolari "
        "(passato remoto: fece, diede/dette, stette, venne, tenne, seppe; congiuntivo: abbia, sia, faccia, "
        "dia, stia). Specifica verbo, persona e tempo NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Qual e\' la forma corretta della terza persona singolare del passato remoto del verbo "fare"?',
          "options": {"A": "fece", "B": "facette", "C": "faciò", "D": "fecette"}, "answer": "A"}]),
    "morph_participio": ("morphology", 0.10,
        "Genera UNA domanda a scelta multipla di morfologia italiana sul PARTICIPIO PASSATO irregolare di un "
        "verbo (spingere->spinto, offrire->offerto, scegliere->scelto, rompere->rotto, dirigere->diretto, "
        "muovere->mosso, cuocere->cotto). Metti l'infinito NELLA domanda. Una sola opzione corretta.",
        [{"question": 'Qual e\' il participio passato corretto del verbo "spingere"?',
          "options": {"A": "spinto", "B": "spingiuto", "C": "spinguto", "D": "spingito"}, "answer": "A"}]),
    "morph_affissi": ("morphology", 0.10,
        "Genera UNA domanda a scelta multipla di morfologia italiana sull'analisi di una parola in "
        "PREFISSO / RADICE / SUFFISSO / DESINENZA, oppure sul riconoscimento di un ALTERATO (diminutivo, "
        "accrescitivo, vezzeggiativo, dispregiativo) o di un DERIVATO. Metti la parola NELLA domanda. "
        "Una sola opzione corretta.",
        [{"question": 'Nella parola "rifare" l\'elemento "ri-" e\':',
          "options": {"A": "un prefisso", "B": "un suffisso", "C": "la radice", "D": "la desinenza"}, "answer": "A"}]),
}

# large seed bank to diversify lexical choices (not obligatory to use)
WORD_SEEDS = (
    "tempo casa correre bello albero mangiare felice citta parlare veloce acqua grande vecchio "
    "pensare strada dormire forte piccolo vivere luce cantare buono freddo sapere monte perdere "
    "chiaro nuovo capire scienza coscienza camicia ciliegia spiaggia valigia farmacia uomo dio "
    "bue ala arma uovo dito braccio tempio attore eroe gallo poeta celebre acre misero salubre "
    "integro cattivo maggiore minore fare dare stare venire tenere sapere spingere offrire "
    "scegliere rompere dirigere muovere cuocere obbligo commozione appartamento efficiente "
    "sufficiente ingegnere calzolaio scienziato psicologo coscienzioso accelerare soqquadro "
    "perche poiche affinche virtu pero benche cosicche giacche nonostante qualcosa qualcuno "
    "acquazzone acquisto nausea garage regime ossigeno beneficio maleficio magnifico terribile "
    "orribile celere agile fertile nobile umile facile difficile amico nemico medico carico "
    "traffico dialogo catalogo obbligo naufragio omicidio proscioglimento asciugamano guancia "
    "mancia freccia goccia treccia roccia faccia caccia pioggia reggia loggia maggio raggio "
    "cuoco fuoco gioco luogo giogo prego lago mago ago strega bottega collega").split()

NORM = re.compile(r"\s+")
JSON_OBJ = re.compile(r"\{.*\}", re.S)


def norm_q(q):
    return NORM.sub(" ", q.strip().lower())


def norm_key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def exact_key(rec):
    opts = [list(o.values())[0] for o in rec["options"]]
    return hashlib.md5((norm_key(rec["question"]) + "|" +
                        "|".join(sorted(norm_key(t) for t in opts))).encode()).hexdigest()


def build_prompt(subtype, rng):
    cat, share, instr, shots = SUBTYPES[subtype]
    ex = "\n".join(json.dumps(s, ensure_ascii=False) for s in shots)
    seed = rng.choice(WORD_SEEDS)
    return (
        "Sei un esperto di grammatica italiana e autore di test scolastici. " + instr +
        "\n\nRegole: esattamente 4 opzioni con chiavi \"A\",\"B\",\"C\",\"D\"; UNA sola risposta corretta; "
        "opzioni brevi, plausibili e non ambigue; la domanda deve essere autoconsistente e verificabile senza "
        "contesto esterno. "
        f"Per variare, ispirati liberamente alla parola-spunto: '{seed}' (non e' obbligatorio usarla). "
        "Rispondi SOLO con un oggetto JSON su una riga con le chiavi esatte: "
        '"question" (stringa), "options" (oggetto con A,B,C,D), "answer" (una tra "A","B","C","D").'
        "\n\nEsempio del formato richiesto:\n" + ex + "\n\nGenera ora UNA nuova domanda diversa:")


@tenacity.retry(retry=tenacity.retry_if_exception_type(Exception),
                wait=tenacity.wait_exponential(multiplier=1, max=8),
                stop=tenacity.stop_after_attempt(8))
def gen_one(client, model, subtype, rng):
    msg = build_prompt(subtype, rng)
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": msg}],
        temperature=0.95, top_p=0.95, max_tokens=400)
    return r.choices[0].message.content


def parse(raw, subtype):
    m = JSON_OBJ.search(raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    q = d.get("question"); opts = d.get("options"); ans = d.get("answer")
    if not isinstance(q, str) or not isinstance(opts, dict) or ans not in ("A", "B", "C", "D"):
        return None
    if set(opts.keys()) != {"A", "B", "C", "D"}:
        return None
    vals = [str(opts[k]).strip() for k in "ABCD"]
    if any(not v for v in vals) or len(set(v.lower() for v in vals)) != 4:
        return None
    if len(q.strip()) < 12:
        return None
    cat = SUBTYPES[subtype][0]
    return {"question": q.strip(),
            "options": [{k: str(opts[k]).strip()} for k in "ABCD"],
            "answer": ans, "category": cat, "source": f"gen_mistral:{subtype}"}


def existing(path):
    seen_q, seen_k, per_cat = set(), set(), {"orthography": 0, "morphology": 0}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            seen_q.add(norm_q(r["question"]))
            seen_k.add(exact_key(r))
            per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
    return seen_q, seen_k, per_cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ortho", type=int, required=True)
    ap.add_argument("--n-morph", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    targets = {"orthography": args.n_ortho, "morphology": args.n_morph}
    subs = list(SUBTYPES)
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    seen_q, seen_k, per_cat = existing(args.out)
    need = {c: max(0, targets[c] - per_cat.get(c, 0)) for c in targets}
    print(f"[gen] out={args.out} have={per_cat} target={targets} need={need}", flush=True)
    if sum(need.values()) == 0:
        print("[gen] nothing to do", flush=True); return

    lock = threading.Lock()
    out = open(args.out, "a")
    # per-category subtype pools + weights
    by_cat = {"orthography": [], "morphology": []}
    for s, (cat, share, _, _) in SUBTYPES.items():
        by_cat[cat].append((s, share))

    total_need = sum(need.values())
    max_attempts = total_need * 6 + 400
    rng_master = random.Random(args.seed)
    seeds = [rng_master.randrange(1 << 30) for _ in range(max_attempts)]

    def attempt(i):
        # pick the category still needing the most items (proportional), then a subtype
        with lock:
            remaining = {c: max(0, targets[c] - per_cat.get(c, 0)) for c in targets}
        if sum(remaining.values()) == 0:
            return
        rng = random.Random(seeds[i])
        cat = rng.choices(list(remaining), weights=[remaining[c] + 1 for c in remaining])[0]
        pool = by_cat[cat]
        subtype = rng.choices([s for s, _ in pool], weights=[w for _, w in pool])[0]
        raw = gen_one(client, args.model, subtype, rng)
        rec = parse(raw, subtype)
        if rec is None:
            return
        qk = norm_q(rec["question"]); ek = exact_key(rec)
        with lock:
            if per_cat.get(rec["category"], 0) >= targets[rec["category"]]:
                return
            if qk in seen_q or ek in seen_k:
                return
            seen_q.add(qk); seen_k.add(ek)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            per_cat[rec["category"]] = per_cat.get(rec["category"], 0) + 1

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(attempt, i) for i in range(max_attempts)]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="gen"):
            _.result()
            if all(per_cat.get(c, 0) >= targets[c] for c in targets):
                break
    out.close()
    print(f"[gen] done per_cat={per_cat} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
