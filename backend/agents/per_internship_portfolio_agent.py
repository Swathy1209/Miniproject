"""
per_internship_portfolio_agent.py — Per-Internship Portfolio Generator
OrchestrAI Autonomous Multi-Agent System
"""

from __future__ import annotations
import logging
import os
import re
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.PerInternshipPortfolioAgent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0) if GEMINI_API_KEY else None

JOBS_FILE = "database/jobs.yaml"
PORTFOLIO_FILE = "database/portfolio.yaml"
USERS_FILE = "database/users.yaml"
PER_INTERNSHIP_INDEX_FILE = "database/per_internship_portfolios.yaml"

DEFAULT_USER_NAME = "Swathy G"
DEFAULT_SKILLS = ["Python", "Machine Learning", "Data Analysis", "SQL", "TensorFlow", "scikit-learn", "FastAPI", "NLP"]

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')[:40]

def _rank_projects_for_job(projects: list[dict], job_skills: list[str]) -> list[dict]:
    job_skill_set = {s.lower() for s in job_skills}
    scored = []
    for project in projects:
        title = project.get("title", project.get("original_name", "")).lower()
        summary = str(project.get("summary", "")).lower()
        tech_str = str(project.get("technologies", "")).lower()
        score = 0
        matched = []
        for skill in job_skill_set:
            if skill in title or skill in summary:
                score += 3
                matched.append(skill)
            elif skill in tech_str:
                score += 2
                matched.append(skill)
        scored.append({**project, "_relevance_score": score, "_matched_skills": list(set(matched))})
    scored.sort(key=lambda x: x["_relevance_score"], reverse=True)
    return scored

def _generate_portfolio_html(company, role, job_skills, projects, user_name, user_skills, skill_gaps, roadmap_steps, cover_letter_link) -> str:
    ranked = _rank_projects_for_job(projects, job_skills)[:4]
    projects_html = ""
    for p in ranked:
        name = p.get("title", p.get("original_name", "Project"))
        desc = p.get("summary", "A technical project.")
        techs = str(p.get("technologies", "Python, ML"))
        url = p.get("github_link", "#")
        matched = p.get("_matched_skills", [])
        tags = "".join(f'<span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);color:#10b981;padding:4px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;display:inline-block;margin:2px">✓ {s.title()}</span>' for s in matched[:3])
        projects_html += f'<div style="background:#1e293b;border-radius:12px;padding:20px;margin-bottom:15px"><h3>{name}</h3><p>{desc}</p><div>{tags}</div><a href="{url}" style="color:#38bdf8;text-decoration:none">View →</a></div>'

    return f"""<!DOCTYPE html><html lang="en"><head><title>{user_name} - {company} Portfolio</title></head>
    <body style="font-family:'Outfit',sans-serif;background:#0f172a;color:white;padding:40px">
        <h1>{user_name} Portfolio for {company}</h1>
        <h2>{role} Role</h2>
        <div style="margin-bottom:30px"><a href="{cover_letter_link}" style="color:#38bdf8">View My Cover Letter</a></div>
        <div id="projects"><h2>Relevant Projects</h2>{projects_html}</div>
    </body></html>"""

def run_per_internship_portfolio_agent() -> list[dict]:
    logger.info("PerInternshipPortfolioAgent: Starting...")
    jobs_data = read_yaml_from_github(JOBS_FILE)
    jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []
    port_data = read_yaml_from_github(PORTFOLIO_FILE)
    projects = port_data.get("projects", []) if isinstance(port_data, dict) else []
    user_data = read_yaml_from_github(USERS_FILE)
    user = user_data.get("user", {}) if isinstance(user_data, dict) else {}
    user_name = user.get("name", DEFAULT_USER_NAME)
    user_skills = user.get("resume_skills", DEFAULT_SKILLS)

    cl_data = read_yaml_from_github("database/cover_letter_index.yaml")
    cl_list = cl_data.get("cover_letters", []) if isinstance(cl_data, dict) else []
    cl_lookup = {(item.get("company"), item.get("role")): item.get("link", "") for item in cl_list if isinstance(item, dict)}

    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://miniproject-bye9.onrender.com")
    index = []
    generated = 0

    for job in jobs[-15:]:  # Process top 15 most recent for specialized portfolios
        company = job.get("company", "Unknown")
        role = job.get("role", "Intern")
        try:
            html = _generate_portfolio_html(company, role, job.get("technical_skills", []), projects, user_name, user_skills, [], [], cl_lookup.get((company, role), ""))
            slug = f"{_slugify(company)}_{_slugify(role)}"
            file_path = f"frontend/portfolio/internships/{slug}.html"
            _, sha = _get_raw_file(file_path)
            _put_raw_file(file_path, html, sha, f"feat: portfolio for {company}")
            pub_url = f"{base_url}/portfolio/internships/{slug}.html"
            index.append({"company": company, "role": role, "portfolio_url": pub_url})
            generated += 1
            logger.info("PerInternshipPortfolioAgent: ✓ %s", company)
        except Exception as exc:
            logger.error("PerInternshipPortfolioAgent: Failed for %s: %s", company, exc)

    if generated > 0:
        write_yaml_to_github(PER_INTERNSHIP_INDEX_FILE, {"per_internship_portfolios": index})
        append_log_entry({"agent": "PerInternshipPortfolioAgent", "action": f"Generated {generated} pages", "status": "success", "timestamp": datetime.now(timezone.utc).isoformat()})
    return index

if __name__ == "__main__":
    run_per_internship_portfolio_agent()
