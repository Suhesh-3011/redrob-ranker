# Redrob Candidate Ranker

Ranks the 100K-candidate pool against the Redrob AI Senior AI/ML Engineer JD.
Local, deterministic, fully explainable — no hosted LLM calls at any point
during ranking.

## Why this architecture

The JD and dataset are explicitly built to punish naive keyword/embedding
matching: keyword-stuffed skill lists, title/skill mismatches, and ~80
"honeypot" profiles with internally impossible data. A pure similarity-search
system falls for all three. An LLM-graded system violates the hackathon's
hard compute constraint (no network calls during the ranking step) and can't
run 100K candidates in the 5-minute budget anyway.

So this is a **two-stage, mostly offline** hybrid:

```
precompute.py  (offline, network OK, unlimited time)
  ├─ feature_extraction.py  → structured, explainable features per candidate:
  │                            evidenced skill matches, disqualifiers,
  │                            honeypot flags, location/behavioral signals
  └─ TF-IDF + TruncatedSVD  → local, offline "semantic" embedding of each candidate's profile text
        ↓
  artifacts/{features.jsonl, embeddings.npy, candidate_ids.json}
        ↓
rank.py  (THE reproducible ranking step — <5min, CPU-only, no network)
  ├─ scoring.py    → blends structured score (65%) + semantic percentile (35%)
  ├─ reasoning.py  → template-based justification from actual extracted facts
  └─ writes submission.csv
```

### The core design decision: evidence over keywords

A candidate's self-reported `skills[].duration_months` is itself just a
claim — a keyword-stuffed profile can list "Embeddings, 18 months" next to
a career history that's entirely brand design and customer support (we found
exactly this candidate in the sample data: `CAND_0000021`). So a skill only
counts as real evidence if it's independently corroborated in the
candidate's career-history prose. Claims found only in the skills array are
capped at a small fraction of credit and the candidate is flagged
`keyword_stuffing_suspect`.

We also strip out hedge-worded claims (`"taking online courses on RAG"`,
`"experimenting with LangChain for side projects"`) before matching — this
directly implements the JD's own disqualifier for recent LangChain-hobbyist
profiles without production depth, even when the exact right keywords
appear.

### Honeypot handling

`feature_extraction.py` runs 8 internal-consistency checks per candidate
(experience-vs-history-duration mismatch, "expert" skills with near-zero
duration, overlapping concurrent roles, duration/date inconsistencies,
education-before-employment, self-rated-vs-assessed skill contradictions,
etc.). Candidates tripping 2+ checks are flagged `is_honeypot_suspect` and
scored down to near-zero. `rank.py` reports the honeypot rate in the final
top 100 on every run so this is checked, not assumed.

### Scoring

`final_score = 0.65 × structured_score + 0.35 × semantic_percentile`,
gated by disqualifier / honeypot multipliers and a behavioral multiplier
(recency, response rate, notice period, interview completion, etc. — see
`redrob_signals_doc`). Structured score dominates deliberately: this dataset
is built to punish similarity-only ranking. Semantic similarity is
percentile-ranked across the whole pool rather than used as a raw cosine
value, since TF-IDF cosine scores cluster in a narrow band across same-domain
text.

Full weight rationale is documented inline in `src/scoring.py` and
`src/jd_spec.py` — every number there is traceable to a specific line in
`job_description.docx`.

### Reasoning column

Generated entirely from the extracted feature dict, never from an LLM —
this is what makes it reliably pass the Stage-4 "no hallucination" check.
Tone (confident / balanced / hedged) is tied to the candidate's own score
tier, and sentence content depends on which facts are actually present for
that candidate, so top-100 reasoning strings aren't templated duplicates.

### Known limitation (disclosed honestly, not hidden)

The "title-chaser" trajectory signal is tenure-length-based (many jobs under
18 months), not true title-escalation-based. It can over-flag people with
short stints for reasons unrelated to title-chasing (layoffs, contract
roles). Given more time this would be replaced with an actual title-level
progression parser.

## Reproduce

```bash
pip install -r requirements.txt

# Step 1 (offline, no network needed at all -- TF-IDF+SVD is fit locally, no model download)
python src/precompute.py --candidates data/candidates.jsonl.gz --out artifacts

# Step 2 (THE reproducible step, <5min, CPU-only, no network)
python src/rank.py --artifacts artifacts --out submission.csv

# Validate format
python validate_submission.py submission.csv
```

`rank.py` prints the disqualified / honeypot-suspect / keyword-stuffing-
suspect counts across the whole pool and the honeypot rate in the final top
100 on every run, so a >10% honeypot rate is visible before you ever submit.

## Repo layout

```
src/
  jd_spec.py            structured JD encoding (must-haves, disqualifiers, location rules)
  feature_extraction.py candidate → structured feature dict
  scoring.py             feature dict + semantic percentile → final score
  reasoning.py            feature dict → 1-2 sentence justification
  precompute.py           offline: features + embeddings for the full pool
  rank.py                 the reproducible ranking step
data/                    candidates.jsonl.gz goes here (gitignored, too large to commit)
artifacts/               precompute.py output (gitignored, regenerate via precompute.py)
validate_submission.py   organizer-provided format validator
submission_metadata.yaml portal metadata mirror
```

## Actual run results (full 100K pool)

```
Candidates loaded:              100,000
Feature extraction time:        ~57s
TF-IDF+SVD fit+transform time:  ~70s
Disqualified (hard gate):       55,237
Honeypot-suspects flagged:      35
Keyword-stuffing-suspects:      4,157
Ranking step wall time:         5.9s   (budget: 300s)
Honeypot rate in final top 100: 0 / 100 (0.0%)   (disqualification threshold: >10%)
validate_submission.py:         PASSED
```

Top-of-list sanity check (manual, not automated): rank 1-15 are Lead/Senior/
Staff AI/ML engineers at real product companies (Razorpay, Paytm, Apple,
Freshworks, Flipkart) with genuinely corroborated embeddings/vector-search/
eval-framework evidence, not skill-list stuffing. Bottom-of-top-100 (rank
90-100) still holds at real ML titles with real evidence, just with an
honestly-stated gap (location, years-outside-band, or one missing must-have)
per candidate — not filler.

## Reasoning tone note

Tone (confident / balanced / hedged) is keyed to **rank position** within
the final top 100, not score-relative-to-the-top-pick. This dataset's
genuine top-100 candidates cluster tightly in absolute score (a narrow,
deep-quality pool, exactly as the JD predicts), so a max-relative
percentile made nearly every row read "confident" on the first pass — that
was caught and fixed during testing (see git history / methodology
summary) before the final run.



## Sandbox

`redrob_ranker_sandbox.ipynb` is the hackathon sandbox environment (per
`submission_spec.md` §10.5) — open it in Google Colab, set `REPO_URL` in the
first code cell to this repo's URL, and run all cells. It clones the actual
repo (not a copy-paste), installs `requirements.txt`, and runs the full
`precompute.py` → `rank.py` → `validate_submission.py` pipeline against the
bundled 50-candidate `data/sample_candidates.json`, since the sandbox only
needs to demonstrate the ranker on a small sample — the full 100K run is
what the grading side reproduces separately via `reproduce_command`.
Verified end-to-end (fresh install, zero network, zero GPU) before delivery.

## AI tool use

Declared honestly in `submission_metadata.yaml`. Used Claude for
architecture discussion, code review, and catching a real bug (the
skill-corroboration flaw described above, found by testing against the
sample data mid-build, not assumed away).
