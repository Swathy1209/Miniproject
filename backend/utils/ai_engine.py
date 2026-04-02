"""
ai_engine.py — OpenAI-Powered AI Functions
OrchestrAI Autonomous Multi-Agent System

Responsibilities:
  - Extract technical skills from raw resume text (GPT-3.5-turbo)
  - Generate learning roadmap from skill gap analysis (GPT-3.5-turbo)
  - Provide keyword-based fallbacks when OpenAI is unavailable
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("OrchestrAI.AIEngine")

OPENAI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Model priority order — using versions known to work with the v1beta/openai shim.
# gemini-1.5-flash: 1500 RPD | gemini-1.5-pro: 50 RPD
GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),  # primary
    "gemini-1.5-flash-8b",                           # fallback 1
    "gemini-1.5-pro",                                # fallback 2 (low quota but high quality)
]

# ── Per-Model Circuit Breaker ──────────────────────────────────────────────
# When a model's daily quota is exhausted, we skip it for the rest of the run.
_EXHAUSTED_MODELS: set[str] = set()

def _is_daily_quota_error(exc: Exception) -> bool:
    """Detect RESOURCE_EXHAUSTED daily quota errors (not transient rate limits)."""
    msg = str(exc)
    # Check for both "quota exceeded" and "limit: 0" which indicates free tier daily exhaustion
    return (
        "RESOURCE_EXHAUSTED" in msg
        or "GenerateRequestsPerDayPerProjectPerModel" in msg
        or "Quota exceeded for metric" in msg
        or ('limit: 0' in msg and '429' in msg)
    )

def _mark_quota_exceeded(model: str) -> None:
    global _EXHAUSTED_MODELS
    if model not in _EXHAUSTED_MODELS:
        _EXHAUSTED_MODELS.add(model)
        logger.warning(
            "AIEngine: 🚨 Model '%s' daily quota exhausted. Switching to fallbacks.", model
        )

def is_all_quota_exhausted() -> bool:
    """Check if all configured Gemini models have exhausted their daily quota."""
    return all(m in _EXHAUSTED_MODELS for m in GEMINI_MODELS)

def safe_llm_call(
    messages: list[dict],
    max_tokens: int = 400,
    temperature: float = 0.5,
    context: str = "",
) -> str | None:
    """
    Central LLM call with:
     - Per-model circuit breaker (skip if daily quota exhausted)
     - Model fallback chain (tries each model before giving up)
     - Retry logic for transient rate limits (429 RPM)
    Returns the response text or None if all models fail.
    """
    if not OPENAI_API_KEY:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0)

    for model in GEMINI_MODELS:
        if model in _EXHAUSTED_MODELS:
            continue

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                err_msg = str(exc)
                
                # 1. Handle Daily Quota Exhaustion
                if _is_daily_quota_error(exc):
                    _mark_quota_exceeded(model)
                    break # Try next model
                
                # 2. Handle Transient Rate Limit (RPM)
                if "429" in err_msg and attempt < max_retries:
                    # Exponential backoff for RPM limits
                    wait_time = 5 * (attempt + 1)
                    logger.warning("AIEngine: %s RPM limited (429) for '%s'. Retrying in %ds...", model, context, wait_time)
                    time.sleep(wait_time)
                    continue

                # 3. Handle other errors
                logger.debug("AIEngine: %s failed for '%s' — %s", model, context, err_msg)
                break

    logger.warning("AIEngine: ❌ ALL MODELS EXHAUSTED for '%s'.", context)
    return None

# ── Known technical skill keywords for fallback extraction ────────────────────
_KNOWN_SKILLS: list[str] = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "R", "Scala", "Kotlin", "Swift", "SQL", "Bash", "Shell",
    # ML / AI
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Reinforcement Learning", "Neural Networks", "LLM", "Generative AI",
    "Transformers", "BERT", "GPT",
    # ML Libraries
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost", "LightGBM",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    # MLOps
    "MLflow", "Kubeflow", "DVC", "BentoML", "Seldon",
    # Data Engineering
    "Apache Spark", "PySpark", "Apache Kafka", "Apache Airflow", "dbt",
    "Hadoop", "Hive", "Databricks", "Snowflake", "BigQuery",
    # Cloud
    "AWS", "GCP", "Azure", "S3", "EC2", "SageMaker", "Lambda",
    "Google Cloud", "Vertex AI", "Azure ML",
    # DevOps / Infra
    "Docker", "Kubernetes", "Terraform", "CI/CD", "GitHub Actions",
    "Jenkins", "Ansible", "Helm",
    # APIs & Frameworks
    "FastAPI", "Flask", "Django", "REST API", "GraphQL", "gRPC",
    "Streamlit", "Gradio", "LangChain", "LlamaIndex",
    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Cassandra", "DynamoDB", "Neo4j",
    # Tools
    "Git", "GitHub", "Linux", "Jupyter", "Power BI", "Tableau",
    "Excel", "HuggingFace", "OpenAI API",
    # Stats
    "Statistics", "Probability", "Data Analysis", "Data Visualization",
    "Feature Engineering", "A/B Testing",
]


def extract_skills_using_ai(resume_text: str) -> list[str]:
    """
    Extract technical skills from resume text using OpenAI GPT-3.5-turbo.

    Falls back to keyword matching if OpenAI is unavailable.

    Args:
        resume_text: Raw text extracted from the resume PDF.

    Returns:
        Sorted, deduplicated list of technical skill strings.
    """
    if not resume_text.strip():
        logger.warning("AIEngine: Empty resume text — returning empty skills list.")
        return []

    prompt_messages = [
        {
            "role": "system",
            "content": "You are a technical resume analyser. Extract technical skills precisely and return them as a comma-separated list only.",
        },
        {
            "role": "user",
            "content": (
                "Extract ONLY the technical skills from the following resume text.\n"
                "Return them as a clean comma-separated list on a single line.\n"
                "Include: programming languages, frameworks, libraries, tools, platforms, "
                "databases, cloud services, ML/AI technologies.\n"
                "Do NOT include soft skills, job titles, or company names.\n"
                "Example output: Python, SQL, Machine Learning, FastAPI, Docker, AWS\n\n"
                f"Resume text:\n{resume_text[:6000]}"
            ),
        },
    ]
    raw = safe_llm_call(prompt_messages, max_tokens=300, temperature=0.1, context="skill extraction")
    if raw:
        skills = [s.strip() for s in raw.split(",") if s.strip()]
        seen: set[str] = set()
        deduped: list[str] = []
        for skill in skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                deduped.append(skill)
        logger.info("AIEngine: Extracted %d skills from resume.", len(deduped))
        return deduped

    # ── Keyword fallback ──────────────────────────────────────────────────────
    return _keyword_extract_skills(resume_text)


def _keyword_extract_skills(text: str) -> list[str]:
    """Fallback: match known skills against resume text (case-insensitive)."""
    found: list[str] = []
    text_lower = text.lower()
    for skill in _KNOWN_SKILLS:
        # Match whole word / phrase
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    logger.info("AIEngine: Keyword fallback found %d skills.", len(found))
    return found


def generate_per_job_roadmap(
    user_skills: list[str],
    job_skills: list[str],
    missing_skills: list[str],
) -> list[str]:
    """
    Generate a concise, prioritised learning roadmap using OpenAI for a specific job.

    Args:
        user_skills:    Skills the user already has.
        job_skills:     Skills required by this specific job.
        missing_skills: Skills required by the job but not in user's profile.

    Returns:
        List of actionable roadmap step strings.
    """
    if not missing_skills:
        return ["No skill gaps detected. You are well-equipped for this role!"]

    raw = safe_llm_call(
        messages=[
            {"role": "system", "content": "You are an expert technical career coach. Give practical, prioritised advice."},
            {"role": "user", "content": (
                f"User skills: {', '.join(user_skills)}\n\n"
                f"Job requires: {', '.join(job_skills)}\n\n"
                f"Missing skills: {', '.join(missing_skills)}\n\n"
                "Generate a concise learning roadmap for this job.\n"
                "Return ONLY the bullet points, one per line, starting with a dash (-)."
            )},
        ],
        max_tokens=400, temperature=0.5,
        context=f"roadmap for {missing_skills[:2]}",
    )
    if raw:
        roadmap = [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]
        logger.info("AIEngine: Generated %d roadmap steps.", len(roadmap))
        return roadmap or _keyword_roadmap(missing_skills)

    return _keyword_roadmap(missing_skills)


def generate_learning_roadmap(
    user_skills: list[str],
    missing_skills: list[str],
) -> list[str]:
    """
    Generate a concise, prioritised learning roadmap using OpenAI.

    Falls back to a rule-based roadmap if OpenAI is unavailable.

    Args:
        user_skills:    Skills the user already has.
        missing_skills: Skills required by jobs but not in user's profile.

    Returns:
        List of actionable roadmap step strings.
    """
    if not missing_skills:
        return ["No skill gaps detected. You are well-equipped for current listings!"]

    raw = safe_llm_call(
        messages=[
            {"role": "system", "content": "You are an expert technical career coach specialising in AI and Data Science. Give practical, prioritised advice."},
            {"role": "user", "content": (
                f"User's current skills: {', '.join(user_skills)}\n\n"
                f"Missing skills required by AI/Data job listings: {', '.join(missing_skills)}\n\n"
                "Generate a concise, prioritised learning roadmap (5-8 bullet points) "
                "for becoming an industry-ready AI/Data Science engineer.\n"
                "Each step should be specific and actionable.\n"
                "Return ONLY the bullet points, one per line, starting with a dash (-).\n"
                "Order from highest-impact to lowest-impact."
            )},
        ],
        max_tokens=400, temperature=0.5,
        context="general roadmap",
    )
    if raw:
        roadmap = [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]
        logger.info("AIEngine: Generated %d roadmap steps.", len(roadmap))
        return roadmap or _keyword_roadmap(missing_skills)

    return _keyword_roadmap(missing_skills)


def _keyword_roadmap(missing_skills: list[str]) -> list[str]:
    """Rule-based fallback roadmap generation."""
    priority = {
        "docker":       "Learn Docker — containerise your ML models and APIs",
        "kubernetes":   "Learn Kubernetes — orchestrate containers at scale",
        "aws":          "Learn AWS (SageMaker, S3, EC2) — cloud deployment for ML",
        "gcp":          "Learn GCP (Vertex AI, BigQuery) — Google cloud ML stack",
        "azure":        "Learn Azure ML — Microsoft enterprise cloud AI platform",
        "fastapi":      "Build FastAPI services — expose ML models as REST APIs",
        "airflow":      "Master Apache Airflow — schedule and monitor data pipelines",
        "spark":        "Learn Apache Spark/ PySpark — large-scale data processing",
        "pytorch":      "Deep-dive PyTorch — for model research and production",
        "tensorflow":   "Learn TensorFlow — scalable model training and serving",
        "mlflow":       "Adopt MLflow — track experiments and manage model lifecycle",
        "langchain":    "Learn LangChain — build LLM-powered applications",
        "huggingface":  "Explore HuggingFace — fine-tune and deploy transformer models",
        "dbt":          "Learn dbt — transform data in the warehouse like an engineer",
        "kafka":        "Learn Apache Kafka — real-time streaming data pipelines",
        "pyspark":      "Learn PySpark — distributed in-memory data processing",
        "streamlit":    "Build Streamlit apps — rapid ML demo and dashboard creation",
        "terraform":    "Learn Terraform — infrastructure-as-code for cloud resources",
        "kubernetes":   "Learn Kubernetes — scale containerised workloads reliably",
    }
    roadmap = []
    for skill in missing_skills:
        key = skill.lower().replace(" ", "").replace("-", "")
        roadmap.append(
            priority.get(key, f"Learn {skill} via official documentation and project practice")
        )
    return roadmap
