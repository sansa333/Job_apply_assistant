# Evidence-Grounded Job Agent Evaluation Dataset v1

## 1. Intended use

This dataset evaluates a job-seeker-side assistant. It must not be used to make autonomous hiring, rejection, compensation or employment decisions.

The evaluation has four independent targets:

1. structured extraction from public historical job descriptions;
2. evidence-grounded resume–job matching;
3. policy-constrained Agent tool trajectories;
4. unsupported-claim and false-submission detection in generated application material.

## 2. Data composition

- `job_snapshots.jsonl`: eight public historical job descriptions with exact company name, job title, source dataset, source URL, content hash and hand-authored expected fields;
- `candidate_profiles.jsonl`: six detailed synthetic candidates with explicit education, experience, projects, skills, location constraints and negative facts;
- `match_annotations.jsonl`: 24 concrete candidate–job labels with rationale and hard gaps;
- `agent_trajectories.jsonl`: ten concrete workflow-policy cases with expected tools, terminal status, terminal stage and next action;
- `generation_validation.jsonl`: eight supported or unsupported generation cases;
- `manifest.json`: counts and SHA-256 fingerprints.

No candidate is a real applicant. `Maya Patel`, `Li Jiawen`, `Oliver Grant`, `Sofia Martinez`, `Noah Williams` and `Amina Yusuf` are synthetic research personas. Their schools, employers, metrics and project histories are part of the synthetic profile and must not be attributed to a real person with the same name.

The job postings are historical snapshots imported from the checked-in `kyosek_jobs.csv`. They do not imply that Arm, British Airways, Visa, Veeva Systems, UiPath, Transdermal Diagnostics, Warner Bros. Discovery or the University of Edinburgh currently have these roles open.

## 3. Splitting and leakage control

- Candidate splits are declared in `candidate_profiles.jsonl`.
- Model selection must use only `development` candidates; final metrics use only `test` candidates.
- Do not create paraphrases of a test candidate in the development split.
- A future larger version must split by both candidate identity and occupation family.
- Retrieval chunks from a test JD may be indexed for inference, but its manual labels and expected fields must not be used for prompt/model tuning.

## 4. Label status

The current match labels are first-pass annotations by the project author. Every row deliberately carries:

```json
{
  "annotation_status": "expert_review_required",
  "annotator_count": 1,
  "adjudication_status": "pending_second_annotator"
}
```

Until the protocol in `ANNOTATION_GUIDE.md` is completed, reports must say “silver labels” or “first-pass labels”, never “HR expert gold labels”.

## 5. Recommended metrics

- extraction: skill micro/macro F1, education exact match, work-mode exact match, requirement count diagnostics;
- matching: Macro-F1, accuracy, Spearman correlation, MRR and nDCG grouped by candidate;
- Agent: exact tool-sequence accuracy, tool-set F1, terminal-state accuracy and next-action accuracy;
- generation validation: finding-code precision, recall and F1;
- operations: P50/P95 latency, mean requirements per JD, mean evidence facts per candidate and failure count.

Every report must include dataset version, file hashes, scoring version and per-case bad cases.

## 6. Limitations

- Eight jobs and six synthetic candidates are sufficient for a deterministic regression suite, not for claiming general labor-market performance.
- The public source is English-heavy and historically dated.
- The rule baseline uses a finite skill lexicon and can miss unseen occupational terminology.
- Synthetic candidates reduce privacy risk but cannot replace evaluation on consented, anonymized real resumes.
- Matching labels can encode annotator bias; disagreements and demographic slices must be reported when the set grows.
