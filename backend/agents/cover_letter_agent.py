"""
cover_letter_agent.py — Cover Letter Generation Agent
OrchestrAI Autonomous Multi-Agent System

Generates a personalized cover letter for each internship.
Uses Gemini 2.0 Flash via OpenAI-compatible endpoint.
Falls back to a template if LLM is unavailable.
"""

import logging
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.CoverLetterAgent")
USERS_FILE = "database/users.yaml"
JOBS_FILE = "database/jobs.yaml"
COVER_LETTER_INDEX_FILE = "database/cover_letter_index.yaml"

# Default user profile for when users.yaml has no data
DEFAULT_USER = {
    "name": "Swathy G",
    "email": os.getenv("EMAIL_USER", ""),
    "resume_skills": [
        "Python", "Machine Learning", "Data Analysis", "SQL",
        "TensorFlow", "scikit-learn", "Pandas", "NumPy",
        "Data Visualization", "Deep Learning", "NLP", "FastAPI"
    ],
    "career_goals": ["Data Engineering Internship", "ML Engineering Internship"],
    "education": "B.Tech in Computer Science"
}

def read_jobs() -> list[dict]:
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    except Exception as exc:
        logger.error("CoverLetterAgent: read_jobs failed - %s", exc)
        return []

def read_user_profile() -> dict:
    try:
        data = read_yaml_from_github(USERS_FILE)
        user = data.get("user", {}) if isinstance(data, dict) else {}
        # Merge with defaults for any missing fields
        merged = {**DEFAULT_USER, **user}
        if not merged.get("resume_skills"):
            merged["resume_skills"] = DEFAULT_USER["resume_skills"]
        return merged
    except Exception:
        return DEFAULT_USER

import time

from backend.utils.ai_engine import safe_llm_call

def generate_cover_letter(job: dict, user: dict) -> str:
    user_name = user.get("name", "Swathy G")
    resume_skills = ", ".join(user.get("resume_skills", DEFAULT_USER["resume_skills"])[:8])
    role = job.get("role", "Intern")
    company = job.get("company", "Company")
    technical_skills = ", ".join(job.get("technical_skills", [])[:6])
    education = user.get("education", "B.Tech in Computer Science")

    prompt = f"""Write a professional, enthusiastic, and fully personalized internship cover letter.

Candidate Name: {user_name}
Education: {education}
Key Skills: {resume_skills}

Target Role: {role}
Company: {company}
Required Skills: {technical_skills}

Instructions:
- Address the hiring manager professionally
- Highlight specific relevant skills that match the role
- Show genuine enthusiasm for the specific company and role
- Mention one concrete way you can add value
- Keep it to exactly 3 paragraphs (150-200 words total)
- Do NOT use any placeholders like [Your Name] or [Date]
- Write in first person, confident and professional tone
"""

    content = safe_llm_call(
        messages=[
            {"role": "system", "content": "You are an expert career counselor. Generate complete, professional cover letters with no placeholders."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7,
        context=f"cover letter:{company[:20]}"
    )

    if content and len(content) > 50:
        return content

    # Fallback template
    return (
        f"Dear Hiring Manager at {company},\n\n"
        f"I am writing to express my strong interest in the {role} position at {company}. "
        f"As a {education} student with hands-on expertise in {resume_skills}, "
        f"I am excited by the opportunity to contribute to your team's work in {technical_skills}.\n\n"
        f"Throughout my academic journey, I have developed a solid foundation in machine learning, data analysis, "
        f"and software development. I am particularly drawn to {company}'s innovative approach and believe my "
        f"technical skills align perfectly with the requirements of this role. I am eager to apply "
        f"my knowledge to solve real-world challenges and grow as a professional.\n\n"
        f"I would welcome the opportunity to discuss how I can contribute to {company}'s mission. "
        f"Thank you for considering my application.\n\n"
        f"Sincerely,\n{user_name}\n"
    )

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')[:40]

def save_cover_letter(company: str, role: str, letter_text: str) -> str:
    file_name = f"{_slugify(company)}_{_slugify(role)}.md"
    file_path = f"cover_letters/{file_name}"
    content = f"# Cover Letter — {company} — {role}\n\n{letter_text}"

    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(file_path, content, sha, f"feat: cover letter for {company} {role} — {ts}")
        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://miniproject-bye9.onrender.com")
        return f"{base_url}/{file_path}"
    except Exception as exc:
        logger.error("CoverLetterAgent: save_cover_letter failed - %s", exc)
        return ""

def run_cover_letter_agent() -> list[dict]:
    """
    Main agent for generating personalized cover letters.
    """
    from backend.utils.ai_engine import is_all_quota_exhausted
    logger.info("CoverLetterAgent: Starting...")
    
    jobs = read_jobs()
    user = read_user_profile()

    if not jobs:
        logger.warning("CoverLetterAgent: No jobs found. Skipping.")
        return []

    # Load existing index to skip duplicates
    existing_data = read_yaml_from_github(COVER_LETTER_INDEX_FILE)
    index = existing_data.get("cover_letters", []) if isinstance(existing_data, dict) else []
    existing_keys = {(s.get("company"), s.get("role")) for s in index}

    # Process last 40 jobs (cover letters are faster)
    jobs_to_process = jobs[-40:]
    new_entries = 0

    for job in jobs_to_process:
        company = job.get("company", "Unknown")
        role = job.get("role", "Intern")

        if (company, role) in existing_keys:
            logger.info("CoverLetterAgent: Skipping %s - %s (exists)", company, role)
            continue
            
        if is_all_quota_exhausted():
            logger.warning("CoverLetterAgent: Quota exhausted - stopping.")
            break

        time.sleep(3)

        try:
            letter = generate_cover_letter(job, user)
            if not letter:
                continue
            link = save_cover_letter(company, role, letter)
            if link:
                index.append({"company": company, "role": role, "link": link})
                new_entries += 1
                logger.info("CoverLetterAgent: ✓ %s", company)
        except Exception as exc:
            logger.error("CoverLetterAgent: Failed for %s: %s", company, exc)

    if new_entries > 0:
        write_yaml_to_github(COVER_LETTER_INDEX_FILE, {"cover_letters": index})
        append_log_entry({
            "agent": "CoverLetterAgent",
            "action": f"Generated {new_entries} new cover letters",
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    return index


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    results = run_cover_letter_agent()
    print(f"\nGenerated {len(results)} cover letters.")
