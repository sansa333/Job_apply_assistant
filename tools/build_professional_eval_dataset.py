from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import settings
from app.knowledge.catalog import JobCatalog


OUTPUT_DIR = settings.data_dir / "eval_dataset" / "job_agent_v1"


JOB_ANNOTATIONS = {
    "job_0adf266dea1dc6d8d1cb3fb2": {
        "company": "Arm",
        "title": "Full Stack Data Scientist",
        "occupation_family": "data_science",
        "expected_skills": ["python", "javascript", "flask", "django", "react", "machine learning", "docker", "kubernetes", "aws", "azure", "gcp"],
        "education": "Bachelor's or Master's degree in Computer Science, Data Science, or related field",
        "work_mode": "hybrid",
    },
    "job_177950a647e91cacdd7e94d3": {
        "company": "British Airways",
        "title": "Data Scientist - Machine Learning Products",
        "occupation_family": "data_science",
        "expected_skills": ["python", "machine learning", "statistics", "data visualization"],
        "minimum_years": 2,
        "education": "Numerate degree focused on data science, machine learning, or related field",
    },
    "job_124a7fd6b9536d9b4371dc82": {
        "company": "Visa",
        "title": "Senior Analyst Credit & Settlement Risk (12 month FTC)",
        "occupation_family": "financial_risk",
        "expected_skills": ["credit risk", "risk management"],
        "work_mode": "hybrid, office 2-3 days per week",
    },
    "job_f7e7cf1579f3baa26723734d": {
        "company": "Veeva Systems",
        "title": "Principal Data Scientist - United Kingdom (Remote)",
        "occupation_family": "applied_nlp",
        "expected_skills": ["python", "machine learning", "nlp", "deep learning", "pytorch", "aws"],
        "work_mode": "remote within the United Kingdom",
    },
    "job_22e2d03b87d19d4a43151008": {
        "company": "UiPath",
        "title": "Senior Principal Applied Scientist",
        "occupation_family": "applied_ai_research",
        "expected_skills": ["python", "machine learning", "deep learning", "nlp", "pytorch"],
    },
    "job_b3589eaeff7e87a5e79da12e": {
        "company": "Transdermal Diagnostics Ltd",
        "title": "Data Engineer",
        "occupation_family": "health_data_engineering",
        "expected_skills": ["machine learning", "statistics", "data visualization", "signal processing", "medical devices", "risk management"],
        "education": "Master's degree in a quantitative or engineering field; PhD plus two years of postdoctoral work or equivalent industrial experience",
        "work_mode": "remote or Bristol, United Kingdom",
    },
    "job_b738213124e06948c5554812": {
        "company": "Warner Bros. Discovery",
        "title": "Broadcast Engineer",
        "occupation_family": "broadcast_engineering",
        "expected_skills": ["broadcast engineering", "networking", "virtualization", "cloud services"],
        "work_mode": "Stockley Park, London; shift pattern including weekends and occasional nights",
    },
    "job_993c1505bbadcba2608f3bf1": {
        "company": "The University of Edinburgh",
        "title": "Postdoctoral Research Fellow",
        "occupation_family": "life_science_research",
        "expected_skills": ["transcriptomics", "proteomics", "microscopy"],
        "education": "PhD submitted or examined in cell biology or a relevant biological/life sciences subject",
        "work_mode": "Edinburgh, fixed-term full-time role; no visa sponsorship stated in historical snapshot",
    },
}


CANDIDATES = [
    {
        "candidate_id": "syn_maya_patel_ml_platform",
        "display_name": "Maya Patel",
        "synthetic": True,
        "occupation_family": "ml_platform",
        "split": "development",
        "resume_text": """Maya Patel\nTarget: Machine Learning / Data Science Engineer\nLocation: Cambridge, United Kingdom; open to hybrid work.\nEducation: MSc Data Science, University of Bristol; BSc Computer Science.\nExperience: 4 years building Python machine-learning products. At Northstar Retail Analytics she deployed demand-forecasting models through FastAPI, Docker and Kubernetes on AWS. She implemented batch and real-time feature pipelines with SQL, Airflow and PostgreSQL, and partnered with React engineers on model-monitoring dashboards.\nProject evidence: Reduced inference P95 latency by 31% after profiling model and API bottlenecks. Built an MLflow experiment registry and production drift alerts.\nSkills: Python, SQL, FastAPI, Flask, React, machine learning, statistics, data visualization, AWS, Docker, Kubernetes, Airflow, PostgreSQL, MLflow.\nConstraints: Does not claim credit-risk underwriting, broadcast engineering, biomedical laboratory work or a PhD.""",
    },
    {
        "candidate_id": "syn_li_jiawen_llm_application",
        "display_name": "Li Jiawen",
        "synthetic": True,
        "occupation_family": "llm_application",
        "split": "development",
        "resume_text": """Li Jiawen\nTarget: LLM Application Engineer\nLocation: Shanghai, China; available for remote collaboration but does not hold UK work authorization.\nEducation: Master of Electronic Information, Shanghai University; Bachelor of Intelligent Science and Technology.\nExperience: 2 years of project and internship experience with Python services and retrieval-augmented generation. Built a FastAPI service using LangChain, Chroma, BM25/RRF retrieval and a BGE reranker. Designed Pydantic tool schemas, evidence citations and request-level audit logs.\nProject evidence: Created a Chinese evaluation set for resume and job-description retrieval; implemented Docker packaging and pytest regression tests.\nSkills: Python, FastAPI, LangChain, RAG, Chroma, Docker, NLP, PyTorch, SQL.\nConstraints: No production full-stack JavaScript experience, no financial underwriting experience and no biomedical PhD.""",
    },
    {
        "candidate_id": "syn_oliver_grant_credit_risk",
        "display_name": "Oliver Grant",
        "synthetic": True,
        "occupation_family": "financial_risk",
        "split": "test",
        "resume_text": """Oliver Grant\nTarget: Credit Risk and Portfolio Analytics\nLocation: London, United Kingdom; available for hybrid office work.\nEducation: MSc Finance, London School of Economics; BSc Economics.\nExperience: 6 years in credit underwriting and counterparty-risk analysis at a UK commercial bank. Managed a portfolio of 180 corporate and non-bank financial institution clients, completed annual reviews, monitored watch lists and designed stress-testing scenarios. Partnered with legal and collateral teams during covenant breaches and restructuring cases.\nProject evidence: Automated portfolio reporting with SQL and Python and introduced early-warning indicators for rating deterioration.\nSkills: credit risk, risk management, SQL, Python, statistics, stakeholder communication.\nConstraints: No machine-learning production deployment, broadcast engineering or laboratory research experience.""",
    },
    {
        "candidate_id": "syn_sofia_martinez_regeneration",
        "display_name": "Sofia Martinez",
        "synthetic": True,
        "occupation_family": "life_science_research",
        "split": "test",
        "resume_text": """Sofia Martinez\nTarget: Postdoctoral Research in Tissue Regeneration\nLocation: Manchester, United Kingdom; has independent UK work authorization.\nEducation: PhD Cell Biology, University of Manchester; thesis on macrophage signalling during skin repair.\nExperience: 5 years of doctoral and postdoctoral laboratory research. Designed murine wound-healing experiments, maintained complete laboratory records and co-authored three peer-reviewed manuscripts.\nProject evidence: Performed immunohistochemistry, PCR, confocal microscopy and primary mouse-cell culture. Analysed single-cell RNA sequencing and spatial transcriptomics data with Python and R; collaborated on a proteomics study.\nSkills: microscopy, transcriptomics, proteomics, Python, statistics, scientific writing, murine models.\nConstraints: No commercial credit-risk, broadcast or production web-platform experience.""",
    },
    {
        "candidate_id": "syn_noah_williams_broadcast",
        "display_name": "Noah Williams",
        "synthetic": True,
        "occupation_family": "broadcast_engineering",
        "split": "test",
        "resume_text": """Noah Williams\nTarget: Broadcast Engineer\nLocation: London, United Kingdom; accepts rotating shifts, weekends and occasional nights.\nEducation: BEng Broadcast and Media Systems.\nExperience: 8 years supporting live football and rugby productions. Provided tier-1 and tier-2 incident response for studios, galleries and post-production systems; maintained technical documentation and coordinated escalation during live events.\nProject evidence: Migrated an HD facility to SMPTE ST 2110 IP video with AES67 audio and NMOS discovery. Troubleshot UHD/HDR signal paths, Dolby Atmos monitoring, replay systems, virtualization, storage and network faults.\nSkills: broadcast engineering, Linux, networking, cloud services, incident management, documentation, project coordination.\nConstraints: No data-science model development, credit underwriting or life-science research.""",
    },
    {
        "candidate_id": "syn_amina_yusuf_data_engineering",
        "display_name": "Amina Yusuf",
        "synthetic": True,
        "occupation_family": "data_engineering",
        "split": "development",
        "resume_text": """Amina Yusuf\nTarget: Data Engineer\nLocation: Bristol, United Kingdom; prefers remote-first or hybrid roles.\nEducation: BSc Software Engineering.\nExperience: 4 years developing healthcare and telemetry data pipelines. Built Python and SQL ingestion services, Spark transformations and Airflow DAGs on AWS; operated Kafka topics, PostgreSQL warehouses and Docker workloads.\nProject evidence: Implemented data-quality checks for wearable-sensor streams and reduced failed daily pipelines by 38%. Worked with data scientists to expose curated features but did not train research models.\nSkills: Python, SQL, Spark, Airflow, Kafka, PostgreSQL, AWS, Docker, data quality.\nConstraints: No PhD, no murine laboratory experience, no financial underwriting and no broadcast engineering.""",
    },
]


MATCH_ANNOTATIONS = [
    ("syn_maya_patel_ml_platform", "job_0adf266dea1dc6d8d1cb3fb2", "high", "Direct Python, Flask/React, ML deployment, cloud and container evidence; relevant MSc.", []),
    ("syn_maya_patel_ml_platform", "job_177950a647e91cacdd7e94d3", "high", "Four years of production ML, Python, analytics and dashboard evidence match the role.", []),
    ("syn_maya_patel_ml_platform", "job_f7e7cf1579f3baa26723734d", "medium", "Strong ML platform evidence but limited multi-domain NLP and life-sciences product evidence.", ["multi-domain NLP", "life-sciences domain"]),
    ("syn_maya_patel_ml_platform", "job_124a7fd6b9536d9b4371dc82", "low", "Same broad analytics family but no credit underwriting, collateral or financial-institution portfolio evidence.", ["credit underwriting", "collateral management"]),
    ("syn_li_jiawen_llm_application", "job_f7e7cf1579f3baa26723734d", "medium", "Direct NLP, Python and PyTorch evidence; insufficient principal-level production ownership and life-sciences domain depth.", ["principal-level experience", "life-sciences domain"]),
    ("syn_li_jiawen_llm_application", "job_0adf266dea1dc6d8d1cb3fb2", "medium", "Python and ML application evidence exists, but full-stack JavaScript and production model deployment are incomplete.", ["JavaScript full stack", "production model deployment"]),
    ("syn_li_jiawen_llm_application", "job_22e2d03b87d19d4a43151008", "low", "Applied LLM engineering does not establish senior-principal research leadership.", ["senior research leadership", "research publication record"]),
    ("syn_li_jiawen_llm_application", "job_b738213124e06948c5554812", "low", "No live broadcast chain, SMPTE ST 2110 or production-support evidence.", ["broadcast chain", "live production support"]),
    ("syn_oliver_grant_credit_risk", "job_124a7fd6b9536d9b4371dc82", "high", "Direct underwriting, portfolio, watch-list, stress-testing and stakeholder evidence for a London hybrid risk role.", []),
    ("syn_oliver_grant_credit_risk", "job_177950a647e91cacdd7e94d3", "low", "Python analytics is present, but no machine-learning model development or deployment evidence.", ["machine-learning production"]),
    ("syn_oliver_grant_credit_risk", "job_0adf266dea1dc6d8d1cb3fb2", "low", "No full-stack or machine-learning product evidence despite Python/SQL overlap.", ["full-stack development", "machine-learning deployment"]),
    ("syn_oliver_grant_credit_risk", "job_993c1505bbadcba2608f3bf1", "low", "No PhD or biological laboratory evidence.", ["cell-biology PhD", "murine models"]),
    ("syn_sofia_martinez_regeneration", "job_993c1505bbadcba2608f3bf1", "high", "Direct PhD, skin-repair, murine model, microscopy, transcriptomics, proteomics and manuscript evidence.", []),
    ("syn_sofia_martinez_regeneration", "job_b3589eaeff7e87a5e79da12e", "medium", "Relevant health data and Python analysis, but limited production data-engineering infrastructure evidence.", ["production data pipelines", "cloud operations"]),
    ("syn_sofia_martinez_regeneration", "job_f7e7cf1579f3baa26723734d", "low", "Life-science knowledge is relevant, but the role requires production NLP pipelines and principal data-science ownership.", ["NLP pipeline", "production deployment"]),
    ("syn_sofia_martinez_regeneration", "job_b738213124e06948c5554812", "low", "No broadcast production or engineering support evidence.", ["broadcast engineering"]),
    ("syn_noah_williams_broadcast", "job_b738213124e06948c5554812", "high", "Direct live-sports, tier support, ST 2110, AES67, NMOS, UHD/HDR and shift availability evidence.", []),
    ("syn_noah_williams_broadcast", "job_0adf266dea1dc6d8d1cb3fb2", "low", "Infrastructure overlap does not provide data-science or full-stack model-product evidence.", ["machine learning", "full-stack development"]),
    ("syn_noah_williams_broadcast", "job_124a7fd6b9536d9b4371dc82", "low", "No credit analysis or financial-services risk evidence.", ["credit risk"]),
    ("syn_noah_williams_broadcast", "job_993c1505bbadcba2608f3bf1", "low", "No biological PhD or laboratory methods.", ["PhD", "laboratory research"]),
    ("syn_amina_yusuf_data_engineering", "job_b3589eaeff7e87a5e79da12e", "medium", "Healthcare telemetry and data-pipeline evidence transfers, but real-time signal processing, medical-device regulations and the advanced-degree condition are incomplete.", ["real-time signal processing", "medical-device regulations", "advanced degree"]),
    ("syn_amina_yusuf_data_engineering", "job_0adf266dea1dc6d8d1cb3fb2", "medium", "Strong backend data stack but limited front-end and model-training/deployment ownership.", ["front-end development", "model training"]),
    ("syn_amina_yusuf_data_engineering", "job_177950a647e91cacdd7e94d3", "medium", "Production data foundations are relevant, but she did not develop ML models.", ["machine-learning model development"]),
    ("syn_amina_yusuf_data_engineering", "job_993c1505bbadcba2608f3bf1", "low", "Healthcare data experience cannot substitute for a PhD and wet-lab methods.", ["PhD", "murine laboratory methods"]),
]


TRAJECTORY_CASES = [
    {
        "case_id": "traj_001_arm_fit_with_profile",
        "description": "Maya Patel requests an Arm Full Stack Data Scientist fit analysis with a stored profile.",
        "policy_input": {"intent": "fit_analysis", "has_candidate_id": True, "has_resume_text": False},
        "expected_tools": ["activate_skill", "retrieve_job", "retrieve_profile", "parse_job_requirements", "align_candidate_evidence", "score_job_fit"],
        "terminal_status": "ok",
        "terminal_stage": "output_validated",
    },
    {
        "case_id": "traj_002_ba_fit_inline_resume",
        "description": "Amina Yusuf supplies resume text for the British Airways role without a candidate ID.",
        "policy_input": {"intent": "fit_analysis", "has_candidate_id": False, "has_resume_text": True},
        "expected_tools": ["activate_skill", "retrieve_job", "parse_job_requirements", "align_candidate_evidence", "score_job_fit"],
        "terminal_status": "ok",
        "terminal_stage": "output_validated",
    },
    {
        "case_id": "traj_003_visa_missing_candidate",
        "description": "Visa risk role is known but no candidate profile or resume is supplied.",
        "policy_input": {"intent": "fit_analysis", "has_candidate_id": False, "has_resume_text": False},
        "expected_tools": ["activate_skill", "retrieve_job"],
        "terminal_status": "needs_candidate_evidence",
        "terminal_stage": "blocked",
        "next_action": "provide_candidate_id_or_resume_text",
    },
    {
        "case_id": "traj_004_arm_unknown_title_without_jd",
        "description": "The user asks for an Arm Compiler Verification Engineer role that is absent from the snapshot and supplies no JD.",
        "policy_input": {"intent": "fit_analysis", "has_jd": False, "job_exists": False, "has_candidate_id": True},
        "expected_tools": ["activate_skill", "retrieve_job"],
        "terminal_status": "job_not_found",
        "terminal_stage": "blocked",
        "next_action": "upload_target_jd",
    },
    {
        "case_id": "traj_005_wbd_package_contacts_missing",
        "description": "Noah Williams requests a Warner Bros. Discovery application package without contact details.",
        "policy_input": {"intent": "application_package", "has_candidate_id": True, "has_all_contacts": False},
        "expected_tools": ["activate_skill", "retrieve_job", "retrieve_profile", "parse_job_requirements", "align_candidate_evidence", "score_job_fit", "generate_materials", "validate_grounded_output"],
        "terminal_status": "generated_not_submitted",
        "terminal_stage": "awaiting_human_confirmation",
        "next_action": "confirm_contact_details_before_sending",
    },
    {
        "case_id": "traj_006_edinburgh_package_valid_contacts",
        "description": "Sofia Martinez requests an Edinburgh application package with confirmed contacts.",
        "policy_input": {"intent": "application_package", "has_candidate_id": True, "has_all_contacts": True},
        "expected_tools": ["activate_skill", "retrieve_job", "retrieve_profile", "parse_job_requirements", "align_candidate_evidence", "score_job_fit", "generate_materials", "validate_grounded_output"],
        "terminal_status": "generated_not_submitted",
        "terminal_stage": "awaiting_human_confirmation",
        "next_action": "review_and_confirm_before_sending",
    },
    {
        "case_id": "traj_007_grounding_validation_failure",
        "description": "A generated Arm cover letter invents a 73% revenue increase and must be blocked for review.",
        "policy_input": {"intent": "application_package", "has_candidate_id": True, "has_all_contacts": True, "generation_valid": False},
        "expected_tools": ["activate_skill", "retrieve_job", "retrieve_profile", "parse_job_requirements", "align_candidate_evidence", "score_job_fit", "generate_materials", "validate_grounded_output"],
        "terminal_status": "validation_failed",
        "terminal_stage": "blocked",
        "next_action": "review_unsupported_claims",
    },
    {
        "case_id": "traj_008_missing_job_identity",
        "description": "The request asks for a tailored package without company or job title.",
        "policy_input": {"intent": "application_package", "has_company": False, "has_job_title": False, "has_candidate_id": True},
        "expected_tools": ["activate_skill"],
        "terminal_status": "needs_job_identity",
        "terminal_stage": "blocked",
        "next_action": "provide_company_and_job_title",
    },
    {
        "case_id": "traj_009_general_career_advice",
        "description": "Li Jiawen asks how to prepare for an LLM application internship without naming a specific employer.",
        "policy_input": {"intent": "general_advice", "has_company": False, "has_job_title": False, "has_jd": False},
        "expected_tools": ["activate_skill"],
        "terminal_status": "completed",
        "terminal_stage": "completed",
    },
    {
        "case_id": "traj_010_veeva_interview_prep",
        "description": "Maya Patel requests interview preparation for Veeva Systems with verified profile evidence.",
        "policy_input": {"intent": "interview_prep", "has_candidate_id": True},
        "expected_tools": ["activate_skill", "retrieve_job", "retrieve_profile", "parse_job_requirements", "align_candidate_evidence", "score_job_fit", "generate_materials", "validate_grounded_output"],
        "terminal_status": "generated_not_submitted",
        "terminal_stage": "awaiting_human_confirmation",
        "next_action": "review_and_confirm_before_sending",
    },
]


GENERATION_CASES = [
    {"case_id": "gen_001_supported_metric", "evidence": ["Reduced inference P95 latency by 31% after profiling model and API bottlenecks."], "text": "Maya reduced inference P95 latency by 31%.", "expected_codes": []},
    {"case_id": "gen_002_invented_metric", "evidence": ["Reduced inference P95 latency by 31% after profiling model and API bottlenecks."], "text": "Maya increased model revenue by 73% and cut latency by 31%.", "expected_codes": ["unsupported_quantified_claim"]},
    {"case_id": "gen_003_false_submission_zh", "evidence": ["The application package is a local draft."], "text": "已完成投递，请等待招聘方回复。", "expected_codes": ["false_submission_claim"]},
    {"case_id": "gen_004_false_submission_en", "evidence": ["The application package is a local draft."], "text": "Application submitted to British Airways.", "expected_codes": ["false_submission_claim"]},
    {"case_id": "gen_005_supported_pipeline_metric", "evidence": ["reduced failed daily pipelines by 38%."], "text": "Amina reduced failed daily pipelines by 38%.", "expected_codes": []},
    {"case_id": "gen_006_invented_years", "evidence": ["4 years developing healthcare and telemetry data pipelines."], "text": "Amina has 8 years of healthcare data-engineering experience.", "expected_codes": ["unsupported_quantified_claim"]},
    {"case_id": "gen_007_safe_draft_status", "evidence": ["Noah accepts rotating shifts."], "text": "The cover letter draft is ready for Noah to review; it has not been submitted.", "expected_codes": []},
    {"case_id": "gen_008_supported_client_count", "evidence": ["Managed a portfolio of 180 corporate and non-bank financial institution clients."], "text": "Oliver managed a portfolio of 180 corporate and NBFI clients.", "expected_codes": []},
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = JobCatalog(settings.job_catalog_path)
    jobs: list[dict] = []
    for job_id, annotation in JOB_ANNOTATIONS.items():
        record = catalog.get(job_id)
        if record is None:
            raise RuntimeError(f"Required public job snapshot is absent from the catalog: {job_id}")
        if record.company_name != annotation["company"] or record.job_title != annotation["title"]:
            raise RuntimeError(f"Catalog identity drift for {job_id}")
        jobs.append(
            {
                "job_id": record.job_id,
                "company_name": record.company_name,
                "job_title": record.job_title,
                "description": record.description,
                "location": record.location,
                "language": record.language,
                "source_dataset": record.source_dataset,
                "source_file": record.source_file,
                "source_url": record.source_url,
                "content_sha256": hashlib.sha256(record.description.encode("utf-8")).hexdigest(),
                "historical_snapshot": True,
                "annotation": annotation,
            }
        )

    matches = [
        {
            "annotation_id": f"match_{index:03d}",
            "candidate_id": candidate_id,
            "job_id": job_id,
            "relevance_label": label,
            "rationale": rationale,
            "hard_gaps": gaps,
            "annotation_status": "expert_review_required",
            "annotator_count": 1,
            "adjudication_status": "pending_second_annotator",
        }
        for index, (candidate_id, job_id, label, rationale, gaps) in enumerate(MATCH_ANNOTATIONS, start=1)
    ]

    write_jsonl(OUTPUT_DIR / "job_snapshots.jsonl", jobs)
    write_jsonl(OUTPUT_DIR / "candidate_profiles.jsonl", CANDIDATES)
    write_jsonl(OUTPUT_DIR / "match_annotations.jsonl", matches)
    write_jsonl(OUTPUT_DIR / "agent_trajectories.jsonl", TRAJECTORY_CASES)
    write_jsonl(OUTPUT_DIR / "generation_validation.jsonl", GENERATION_CASES)

    file_hashes = {}
    for name in (
        "job_snapshots.jsonl",
        "candidate_profiles.jsonl",
        "match_annotations.jsonl",
        "agent_trajectories.jsonl",
        "generation_validation.jsonl",
    ):
        file_hashes[name] = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
    manifest = {
        "dataset_name": "evidence_grounded_job_agent_eval",
        "version": "1.0.0",
        "created_at": "2026-08-27",
        "language_mix": ["en", "zh"],
        "counts": {
            "public_historical_job_snapshots": len(jobs),
            "synthetic_candidate_profiles": len(CANDIDATES),
            "human_match_annotations": len(matches),
            "agent_trajectory_cases": len(TRAJECTORY_CASES),
            "generation_validation_cases": len(GENERATION_CASES),
        },
        "privacy": "All candidate identities and biographies are synthetic. Company and job names come from public historical job snapshots.",
        "label_status": "Match labels are first-pass annotations and must not be reported as gold until second annotation and adjudication are complete.",
        "file_sha256": file_hashes,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
