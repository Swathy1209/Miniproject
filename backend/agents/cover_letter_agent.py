"""
cover_letter_agent.py — Cover Letter Generation Agent
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Read database/jobs.yaml
  2. Read database/users.yaml
  3. Generate cover letters per job via OpenAI
  4. Save each markdown cover letter to GitHub cover_letters/ folder
  5. Update database/cover_letter_index.yaml
  6. Log to database/agent_logs.yaml
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.CoverLetterAgent")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

JOBS_FILE = "database/jobs.yaml"
USERS_FILE = "database/users.yaml"
COVER_LETTER_INDEX_FILE = "database/cover_letter_index.yaml"
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "orchestrai-db")

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
        return data.get("user", {}) if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("CoverLetterAgent: read_user_profile failed - %s", exc)
        return {}

def generate_cover_letter(job: dict, user: dict) -> str:
    if not openai_client:
        logger.error("CoverLetterAgent: OpenAI API Key missing.")
        return ""
        
    user_name = user.get("name", "Applicant")
    resume_skills = ", ".join(user.get("resume_skills", []))
    role = job.get("role", "Intern")
    company = job.get("company", "Company")
    technical_skills = ", ".join(job.get("technical_skills", []))
    
    prompt = f"""Generate a professional internship cover letter.
Candidate Name: {user_name}
Candidate Skills: {resume_skills}
Job Role: {role}
Company: {company}
Required Skills: {technical_skills}

Make it personalized, professional, enthusiastic.
Length: 150–200 words.
Do not include placeholders.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert career counselor generating internship cover letters without placeholders."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("CoverLetterAgent: OpenAI generation failed for %s - %s", company, exc)
        return (
            f"Dear Hiring Manager at {company},\n\n"
            f"I am writing to express my enthusiastic interest in the {role} position. "
            f"With a strong foundation in {resume_skills}, I am well-prepared to contribute to your team. "
            f"I am eager to apply my technical background to real-world challenges at {company}.\n\n"
            f"Sincerely,\n{user_name}\n"
        )

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def save_cover_letter_to_github(company: str, role: str, letter_text: str) -> str:
    file_name = f"{_slugify(company)}_{_slugify(role)}.md"
    file_path = f"cover_letters/{file_name}"
    
    content = f"# Cover Letter – {company} – {role}\n\n{letter_text}"
    
    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(
            file_path,
            content,
            sha,
            f"feat: generate cover letter for {company} {role} — {ts}"
        )
        
        # Build public URL
        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
        return f"{base_url}/{file_path}"
    except Exception as exc:
        logger.error("CoverLetterAgent: save_cover_letter_to_github failed - %s", exc)
        return ""

def update_cover_letter_index(index_data: list[dict]) -> bool:
    try:
        return write_yaml_to_github(COVER_LETTER_INDEX_FILE, {"cover_letters": index_data})
    except Exception as exc:
        logger.error("CoverLetterAgent: update_cover_letter_index failed - %s", exc)
        return False

def log_agent_activity(action: str, status: str = "success") -> None:
    try:
        append_log_entry({
            "agent": "CoverLetterAgent",
            "action": action,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except:
        pass

def run_cover_letter_agent() -> list[dict]:
    logger.info("CoverLetterAgent: Starting...")
    jobs = read_jobs()
    user = read_user_profile()
    
    if not jobs or not user:
        logger.warning("CoverLetterAgent: Missing jobs or user profile.")
        log_agent_activity("Skipped cover letter generation - missing data", "partial")
        return []

    index_data = []
    
    for job in jobs:
        company = job.get("company", "Unknown")
        role = job.get("role", "Unknown")
        
        letter_text = generate_cover_letter(job, user)
        if letter_text:
            github_link = save_cover_letter_to_github(company, role, letter_text)
            if github_link:
                index_data.append({
                    "company": company,
                    "role": role,
                    "link": github_link
                })
                
    if index_data:
        update_cover_letter_index(index_data)
        log_agent_activity(f"Generated cover letters for {len(index_data)} internships")
        
    logger.info("CoverLetterAgent: Completed successfully.")
    return index_data

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_cover_letter_agent()
    print("CoverLetterAgent finished generated index:", result)
