Pre-registration: held-out evaluation v3

Status: written and committed BEFORE any model was run on this set. Date: 2026-10-17 Author: Mike Keror

1. What is frozen
Item	Value
Eval set	eval/qa_pairs_v3.jsonl
sha256 (first 16 hex)	95ddd72d65bffeca
Questions	236
Pipeline code	branch ablation-frozen, tag pipeline-frozen-v1
Corpus	Chroma collection policy_docs (3,414 chunks), unchanged

Strata (final counts):

Stratum	n
ruling_named	65
ruling_unnamed	52
statute	23
noanswer	56
calc	40

Of the 117 ruling questions, 69 come from rulings never used in training or earlier evaluation (unseen), and 48 from rulings that were (seen).

After this commit, none of the following may change: the questions, references or gold values; the pipeline code; the prompts; the judge model and its prompt; the analysis plan below. Any deviation is reported as a deviation, not silently applied.

2. Why this set exists

The earlier 62-question set was used while tuning the pipeline, so its pipeline scores are optimistic. This set was built after the pipeline was frozen, and was reviewed by hand row by row.

3. How the set was built
Candidates: 276, generated from corpus passages by eval/build_eval_v3.py generate (seed 20261014).
Review: every candidate was given keep / fix / drop by hand. 40 were dropped, leaving 236. Automatic flags (eval/v3_precheck.py) were used to triage, never to decide.
Calc gold values were computed by hand, not by the router, to avoid circularity. Disagreements between hand and router values are listed in section 9.
Three different models filled three different roles: the training teacher (gpt-oss-20b), the question writer (qwen/qwen3.8-27b), and the judge (gpt-5.4-mini).
4. Systems under test

All run through the same frozen pipeline and the same generation settings.

Base: Qwen2.5-7B-Instruct
SFT (epoch 1 and epoch 2 adapters; the epoch is chosen on the old validation set, not on v3)
SFT + DPO
Pipeline ablations on the base model: DISABLE_CALCULATOR, DISABLE_CASE_SCOPE, CONTEXT_SNIPPET_CHARS reduced, and all three combined.
5. Hypotheses and outcomes

Primary metric: judged correctness per question (0/1), scored by the frozen judge.

Confirmatory (one test each; results reported whatever they show):

H1. Removing any one pipeline component lowers overall accuracy. Expected direction from the earlier set: each component worth roughly 13 to 26 points.
H2. SFT changes accuracy on the unseen-ruling stratum relative to base. Earlier estimate: +1.6 points, interval [-3.2, +6.5]. No directional prediction beyond "small".
H3. DPO changes accuracy on the unseen-ruling stratum relative to SFT. Earlier estimate: 0.0, interval [-4.8, +4.8].

Exploratory (labelled as such in the paper): per-stratum effects other than the above, calc and noanswer breakdowns, seen versus unseen gaps, error-type analysis.

6. Analysis plan
Unit of resampling is the case, not the question. Ruling questions from the same case are correlated, so I use a cluster bootstrap by the case field, 10,000 resamples, 95% percentile intervals. Non-ruling strata are resampled by question.
Every comparison is paired (same questions for both systems). Report the difference with its interval, not only p-values.
Report unseen-ruling, seen-ruling, statute, noanswer and calc separately. A pooled overall number is given, but no conclusion rests on it alone.
An effect counts as supported only if its interval excludes zero. An interval that includes zero is reported as "not distinguishable from zero at this sample size", with the interval, not as "no effect".
Multiple comparisons: three confirmatory hypotheses are tested; I report all intervals unadjusted and state that clearly.
7. Known limitations (stated in advance)
The statute stratum cannot be truly unseen: the statute text is in the corpus and was used in training. It is reported separately and only claims "held-out questions", not "held-out knowledge".
Only 14 rulings are unseen. Unseen-ruling intervals will be wide because the effective sample is nearer the number of cases than 69.
The statute stratum has 23 questions (under my 25 target). Its interval will be wide, and I will say so.
Questions were written by an LLM and checked by one person. Wording that favours or disfavours particular systems is possible.
The judge is itself an LLM, so judging noise is not zero. A sample of judged answers is re-checked by hand (section 8).
8. Judge checking
Judge: gpt-5.4-mini with the unchanged judge prompt from the earlier evaluation.
After scoring, I hand-check a random sample of 30 judged answers (seed recorded in the results file) and report the agreement rate.
9. Calc hand-versus-router disagreements
None. All 40 hand-computed values matched the router’s output (tolerance 0.5 KES). Hand values were entered before the router’s proposals were compared.

10. Procedure order
Commit this file together with eval/qa_pairs_v3.jsonl. Tag the commit prereg-v3.
Only then run any model on v3.
Publish results with the tag and hash so the order can be checked.
## 11. Tag history and corpus fingerprint

`prereg-v3` (commit 3d53f78) froze the dataset; PREREG.md in that commit was empty because of a staging slip. The text was committed afterwards, before any model was run. `prereg-v3.1` marks the final text. The dataset file is identical at both tags.

Corpus fingerprint (3414 chunks, sha256[:16] of chunk ids and text): <HASH>

## 12. Disclosures

- Section 11 refers to a tag `prereg-v3.1`. That tag was not created; `prereg-v3.2` is the final pre-registration.
- Section 4 was amended before `prereg-v3.2` and before any system under test was run: ablations are on the base model (matching the earlier study), not "the best of the above".
- On 2026-10-17 at about 15:18 (after `prereg-v3`, before `prereg-v3.2`) I ran a five-question harness check (one question per stratum) with gpt-oss-20b on Groq and the harness's default judge (qwen2.5:7b). It was only to confirm the harness reads the v3 file. The output was deleted and is not reported. The tag message on `prereg-v3.2` ("no model run before this tag") should be read as "no system under test".
