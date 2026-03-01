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

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

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

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Truncate to ~6000 chars to stay within token limits
            truncated = resume_text[:6000]

            prompt = (
                "Extract ONLY the technical skills from the following resume text.\n"
                "Return them as a clean comma-separated list on a single line.\n"
                "Include: programming languages, frameworks, libraries, tools, platforms, "
                "databases, cloud services, ML/AI technologies.\n"
                "Do NOT include soft skills, job titles, or company names.\n"
                "Example output: Python, SQL, Machine Learning, FastAPI, Docker, AWS\n\n"
                f"Resume text:\n{truncated}"
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical resume analyser. Extract technical skills "
                            "precisely and return them as a comma-separated list only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,  # Low temperature for deterministic extraction
            )

            raw = response.choices[0].message.content.strip()
            # Parse comma-separated response
            skills = [s.strip() for s in raw.split(",") if s.strip()]
            # Remove duplicates preserving order, deduplicate case-insensitively
            seen: set[str] = set()
            deduped: list[str] = []
            for skill in skills:
                if skill.lower() not in seen:
                    seen.add(skill.lower())
                    deduped.append(skill)

            logger.info("AIEngine: OpenAI extracted %d skills from resume.", len(deduped))
            return deduped

        except Exception as exc:
            logger.warning(
                "AIEngine: OpenAI skill extraction failed (%s) — using keyword fallback.", exc
            )

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

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            prompt = (
                f"User skills: {', '.join(user_skills)}\n\n"
                f"Job requires: {', '.join(job_skills)}\n\n"
                f"Missing skills: {', '.join(missing_skills)}\n\n"
                "Generate a concise learning roadmap for this job.\n"
                "Return ONLY the bullet points, one per line, starting with a dash (-)."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert technical career coach. Give practical, prioritised advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.5,
            )

            raw = response.choices[0].message.content.strip()
            roadmap = [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]
            logger.info("AIEngine: OpenAI generated %d roadmap steps for job.", len(roadmap))
            return roadmap if roadmap else _keyword_roadmap(missing_skills)

        except Exception as exc:
            logger.warning("AIEngine: OpenAI roadmap failed (%s) — using fallback.", exc)

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

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            prompt = (
                f"User's current skills: {', '.join(user_skills)}\n\n"
                f"Missing skills required by AI/Data job listings: {', '.join(missing_skills)}\n\n"
                "Generate a concise, prioritised learning roadmap (5-8 bullet points) "
                "for becoming an industry-ready AI/Data Science engineer.\n"
                "Each step should be specific and actionable.\n"
                "Return ONLY the bullet points, one per line, starting with a dash (-).\n"
                "Order from highest-impact to lowest-impact."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert technical career coach specialising in "
                            "AI and Data Science. Give practical, prioritised advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.5,
            )

            raw = response.choices[0].message.content.strip()
            roadmap = [line.lstrip("- ").strip() for line in raw.splitlines() if line.strip()]
            logger.info("AIEngine: OpenAI generated %d roadmap steps.", len(roadmap))
            return roadmap if roadmap else _keyword_roadmap(missing_skills)

        except Exception as exc:
            logger.warning("AIEngine: OpenAI roadmap failed (%s) — using fallback.", exc)

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
