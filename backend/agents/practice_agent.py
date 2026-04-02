"""
practice_agent.py — Interview Practice Portal Agent
OrchestrAI Autonomous Multi-Agent System
"""

import logging
import os
import re
import time
import yaml
import json
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
)

# Import shared AI engine
from backend.utils.ai_engine import safe_llm_call, is_all_quota_exhausted

logger = logging.getLogger("OrchestrAI.PracticeAgent")

# ──────────────────────────────────────────────────────────────────────────────
# Storage Configuration
# ──────────────────────────────────────────────────────────────────────────────
JOBS_FILE = "database/jobs.yaml"
USER_FILE = "database/users.yaml"
SKILL_GAPS_FILE = "database/skill_gap_per_job.yaml"
RESUME_TEXT_FILE = "database/resume_extracted.txt"
PRACTICE_SESSIONS_FILE = "database/practice_sessions.yaml"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "_", text).strip("_")

def read_jobs() -> list[dict]:
    data = read_yaml_from_github(JOBS_FILE)
    return data.get("jobs", []) if isinstance(data, dict) else []

def read_user_profile() -> dict:
    data = read_yaml_from_github(USER_FILE)
    return data.get("user", {}) if isinstance(data, dict) else []

def load_resume_text() -> str:
    try:
        content, _ = _get_raw_file(RESUME_TEXT_FILE)
        return content if content else ""
    except:
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# Real-Time Interactive Functions (Required by API)
# ──────────────────────────────────────────────────────────────────────────────

def validate_company_role(company: str, role: str) -> bool:
    """Check if company/role exist in jobs database."""
    jobs = read_jobs()
    for j in jobs:
        if j.get("company", "").strip().lower() == company.strip().lower() and j.get("role", "").strip().lower() == role.strip().lower():
            return True
    return False

def generate_interview_response(company: str, role: str, user_input: str) -> dict:
    """Real-time AI response for a practice question."""
    prompt = (
        f"You are a Senior Interviewer at {company}. The candidate is interviewing for: {role}.\n"
        f"Candidate asked/inputted: {user_input}\n\n"
        "Provide a response with four JSON components:\n"
        "1. professional_answer: Expert-level response.\n"
        "2. practice_version: Simplified version for the candidate.\n"
        "3. confidence_tips: 3 short tips.\n"
        "4. detected_language: English or Tamil.\n\n"
        "Return clean JSON only."
    )
    raw = safe_llm_call([{"role": "user", "content": prompt}], max_tokens=1000, context=f"AI Coaching {company}")
    try:
        json_str = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(json_str)
        return data
    except:
        return {
            "professional_answer": "Our AI engine is processing other requests. Please try again.",
            "practice_version": "Wait and retry.",
            "confidence_tips": ["Stay steady.", "Breathe deeply."],
            "detected_language": "English"
        }

def log_interview_interaction(company: str, role: str, user_input: str):
    """Log real-time interaction."""
    append_log_entry({
        "agent": "PracticeAgent",
        "action": f"Real-time coaching: {company} - {role}",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

# ──────────────────────────────────────────────────────────────────────────────
# Batch Portal Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_interview_qa(company: str, role: str, tech_skills: list[str], resume_text: str, user_skills: list[str]) -> list[dict]:
    prompt = (
        f"You are a Senior Technical Interviewer at {company}. Role: {role}.\n"
        "Generate 10 Q&A pairs (JSON list) with keys 'question' and 'answer'."
    )
    raw = safe_llm_call([{"role": "user", "content": prompt}], max_tokens=1000, context=f"Q&A {company}")
    try:
        json_str = re.sub(r"```json|```", "", raw).strip()
        return json.loads(json_str)
    except:
        return []

def _render_practice_html(company, role, qa_pairs, hr_intro) -> str:
    html = f"""<!DOCTYPE html><html><body style="font-family:sans-serif;background:#0f172a;color:white;padding:40px">
        <h1 style="color:#38bdf8">{role} at {company}</h1>
        <div style="background:#1e293b;padding:20px;border-radius:12px;margin:20px 0"><h2>HR Introduction</h2><p>{hr_intro}</p></div>
        <div style="background:#1e293b;padding:20px;border-radius:12px"><h2>Technical Q&A</h2>
        <ul>{"".join(f"<li><b>Q: {q.get('question')}</b><p>A: {q.get('answer')}</p></li>" for q in qa_pairs[:5])}</ul></div>
    </body></html>"""
    return html

def save_practice_html_to_github(company: str, role: str, html_content: str) -> str:
    file_name = f"{_slugify(company)}_{_slugify(role)}.html"
    file_path = f"frontend/practice/{file_name}"
    try:
        _, sha = _get_raw_file(file_path)
        _put_raw_file(file_path, html_content, sha, f"feat: practice portal for {company}")
        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://miniproject-bye9.onrender.com")
        return f"{base_url}/practice/{file_name}"
    except:
        return ""

def run_practice_agent() -> list[dict]:
    logger.info("PracticeAgent: Starting batch run...")
    jobs = read_jobs()
    user = read_user_profile()
    resume_text = load_resume_text()
    
    existing = read_yaml_from_github(PRACTICE_SESSIONS_FILE)
    keys = {(s.get("company"), s.get("role")) for s in existing.get("practice_sessions", []) if isinstance(existing, dict)}
    
    sessions = []
    for job in jobs[-30:]:
        company = job.get("company", "Unknown")
        role = job.get("role", "Intern")
        if (company, role) in keys or is_all_quota_exhausted(): continue
        
        try:
            qa = generate_interview_qa(company, role, job.get("technical_skills", []), resume_text, user.get("resume_skills", []))
            html = _render_practice_html(company, role, qa, "Focus on your technical impact.")
            link = save_practice_html_to_github(company, role, html)
            if link: sessions.append({"company": company, "role": role, "practice_link": link})
        except: pass

    if sessions:
        old_data = existing.get("practice_sessions", []) if isinstance(existing, dict) else []
        write_yaml_to_github(PRACTICE_SESSIONS_FILE, {"practice_sessions": old_data + sessions})
    return sessions

if __name__ == "__main__":
    run_practice_agent()
