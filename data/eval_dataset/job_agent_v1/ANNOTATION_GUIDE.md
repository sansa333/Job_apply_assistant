# Resume–Job Annotation and Adjudication Guide

## 1. Annotation unit

One unit is one exact `(candidate_id, job_id)` pair. Annotators must read the complete candidate profile and complete historical JD. Company reputation, school prestige, age, gender, inferred ethnicity and writing style are out of scope.

## 2. Evidence matrix procedure

1. Copy each JD requirement into a separate row without paraphrasing away numbers, work authorization or mandatory degrees.
2. Mark the requirement as `must_have`, `preferred` or `contextual` using the JD wording.
3. Assign a category: technical skill, responsibility, experience, education, domain, language, location/work mode, soft skill or other.
4. Search only the candidate profile for support.
5. Record an exact evidence quotation and source section. Do not infer a skill from a job title alone.
6. Mark support:
   - `direct`: the profile explicitly demonstrates the skill, responsibility or hard condition;
   - `partial`: related evidence exists but scale, recency, depth or exact condition is incomplete;
   - `missing`: no supporting evidence exists;
   - `contradicted`: the profile explicitly conflicts with the condition.
7. Record every missing or contradicted must-have as a hard gap.

## 3. Pair-level relevance label

- `high`: most weighted requirements are directly supported and there is no contradicted must-have;
- `medium`: several important requirements are supported, but at least one major depth/domain gap or an incompletely supported must-have remains;
- `low`: the occupation family is substantially different, or multiple must-haves lack evidence;
- `insufficient_information`: the resume or JD is too incomplete to decide. This label must not be silently converted to `low`.

The label is not an employment recommendation or probability of hire.

## 4. Hard-negative taxonomy

Annotators must tag at least one reason when a pair is intentionally difficult:

- `skill_overlap_wrong_domain`: shared tools but wrong business/scientific domain;
- `title_overlap_wrong_seniority`: similar title but insufficient level of ownership;
- `analytics_without_ml`: Python/SQL analytics without model development;
- `ml_without_domain`: strong ML but missing regulated/domain knowledge;
- `transferable_infrastructure_only`: cloud/data infrastructure overlaps but core occupation differs;
- `hard_condition_conflict`: work authorization, location, degree or shift constraint conflicts.

## 5. Two-pass annotation

- Annotator A and B label independently and cannot see each other's pair label.
- Calculate Cohen's kappa for the four-way label and agreement for every requirement support value.
- Adjudicate all pair-label disagreements and all must-have disagreements with a third reviewer.
- Preserve pre-adjudication labels; do not overwrite them.
- The gold release must record reviewer role, annotation date, guideline version and adjudication note without publishing personal identities.

Release thresholds:

- pair-label Cohen's kappa at least 0.70;
- at least 95% of must-have rows adjudicated;
- no row with missing rationale or missing evidence quote;
- every source URL, hash and license note present.

## 6. Quality-control examples

For `Oliver Grant × Visa / Senior Analyst Credit & Settlement Risk`, “managed 180 corporate and NBFI clients” is direct portfolio evidence. It does not prove employment at Visa and must not be rewritten as such.

For `Sofia Martinez × University of Edinburgh / Postdoctoral Research Fellow`, the PhD, murine models, microscopy, transcriptomics and proteomics are direct evidence. Her Python experience alone would not justify a high label.

For `Noah Williams × Arm / Full Stack Data Scientist`, Linux and cloud exposure are transferable infrastructure only. They cannot compensate for missing machine-learning and full-stack product evidence.

## 7. Prohibited shortcuts

- Do not label by keyword count alone.
- Do not treat “familiar with” as equivalent to production ownership.
- Do not invent years of experience by adding overlapping dates without checking them.
- Do not infer protected attributes.
- Do not use an LLM label as the final human annotation.
- Do not report first-pass labels as expert gold data.
