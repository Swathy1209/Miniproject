"""
resume_optimization_agent.py — Resume Optimization Agent
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Extract skills from resume (via PDF or fallback to users.yaml).
  2. Load internships from database/jobs.yaml.
  3. Compare skills to generate Missing Skills.
  4. Generate Resume Improvement Suggestions using OpenAI (model: gpt-4o-mini).
  5. Generate Optimized Resume Version (Markdown).
  6. Save optimized resumes to GitHub repo (optimized_resumes/).
  7. Update database/resume_optimizations.yaml.
  8. Return results.
"""

from __future__ import annotations

import logging
import os
import re
import base64
import requests
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
    _auth_headers,
    GITHUB_BRANCH,
    GITHUB_USERNAME,
    GITHUB_REPO
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.ResumeOptimizationAgent")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

JOBS_FILE = "database/jobs.yaml"
USERS_FILE = "database/users.yaml"
OPTIMIZATIONS_FILE = "database/resume_optimizations.yaml"

def download_resume_pdf(repo_path: str = "resumes/swathiga_resume.pdf", local_path: str = "temp_resume.pdf") -> bool:
    try:
        repo_slug = GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
        url = f"https://api.github.com/repos/{repo_slug}/contents/{repo_path}"
        resp = requests.get(url, headers=_auth_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
        if resp.status_code == 200:
            content_b64 = resp.json().get("content", "")
            with open(local_path, "wb") as f:
                f.write(base64.b64decode(content_b64))
            logger.info("ResumeOptimizationAgent: Downloaded resume from %s", repo_path)
            return True
        else:
            logger.warning("ResumeOptimizationAgent: Could not download resume from %s (Status: %d)", repo_path, resp.status_code)
            return False
    except Exception as exc:
        logger.error("ResumeOptimizationAgent: download_resume_pdf failed - %s", exc)
        return False

def extract_skills_from_pdf(local_path: str = "temp_resume.pdf") -> list[str]:
    """Extract text from PDF using PyMuPDF and use OpenAI (or fallback) to extract skills."""
    if not os.path.exists(local_path):
        return []
    
    try:
        doc = fitz.open(local_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if not openai_client:
            logger.warning("OpenAI Key missing; cannot extract skills from PDF text dynamically. Returning keywords fallback.")
            return [] 

        prompt = f"Extract a comma-separated list of technical skills, tools, and technologies from the following resume text:\n\n{text}"
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2
        )
        skills_str = response.choices[0].message.content.strip()
        skills = [s.strip() for s in skills_str.split(",") if s.strip()]
        return skills
    except Exception as exc:
        logger.error("ResumeOptimizationAgent: extract_skills_from_pdf failed - %s", exc)
        return []

def get_fallback_skills() -> list[str]:
    """Fallback to reading users.yaml if PDF extraction fails."""
    try:
        data = read_yaml_from_github(USERS_FILE)
        skills = data.get("user", {}).get("resume_skills", [])
        return [str(s) for s in skills] if isinstance(skills, list) else []
    except:
        return []

def read_jobs() -> list[dict]:
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    except Exception as exc:
        logger.error("ResumeOptimizationAgent: read_jobs failed - %s", exc)
        return []

def generate_suggestions(job: dict, resume_skills: list[str], missing_skills: list[str]) -> list[str]:
    company = job.get("company", "Company")
    role = job.get("role", "Role")
    
    if not openai_client:
        return [f"Add a project showcasing {s}" for s in missing_skills]

    try:
        prompt = (
            f"Based on the resume skills and job requirements, generate resume improvement suggestions to "
            f"increase chances of selection.\n\n"
            f"Job Role: {role}\n"
            f"Company: {company}\n"
            f"Resume Skills: {', '.join(resume_skills)}\n"
            f"Missing Skills: {', '.join(missing_skills)}\n\n"
            f"Return a clean bulleted list of 3-5 concise, specific suggestions. "
            f"Do not include conversational filler."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume optimizer. Return ONLY the bullet points."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        suggestions = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                suggestions.append(line.lstrip("-* ").strip())
        if not suggestions:
            return [content]
        return suggestions
    except Exception as exc:
        logger.warning(f"ResumeOptimizationAgent: OpenAI suggestion fallback used for {company} - {exc}")
        if missing_skills:
            return [f"Add a project showcasing {s}" for s in missing_skills]
        return ["Highlight your existing skills matching the role requirements.", "Tailor your project descriptions to the company operations."]

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def save_optimized_resume_to_github(company: str, role: str, suggestions: list[str], missing_skills: list[str]) -> str:
    file_name = f"{_slugify(company)}_{_slugify(role)}.md"
    file_path = f"optimized_resumes/{file_name}"
    
    sugg_bullets = "\n".join(f"- {s}" for s in suggestions)
    content = f"# Optimized Resume for {company} {role}\n\n## Recommended Additions\n\n{sugg_bullets}\n"
    
    if missing_skills:
        content += f"\n## Suggested Project to Add\n\nBuild a project focusing on: {', '.join(missing_skills)}\n"
    
    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(
            file_path,
            content,
            sha,
            f"feat: generate optimized resume for {company} {role} — {ts}"
        )
        
        repo_slug = GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
        github_link = f"https://github.com/{repo_slug}/blob/main/{file_path}"
        return github_link
    except Exception as exc:
        logger.error("ResumeOptimizationAgent: save_optimized_resume_to_github failed - %s", exc)
        return ""

def update_optimizations_yaml(data: list[dict]) -> bool:
    try:
        return write_yaml_to_github(OPTIMIZATIONS_FILE, data)
    except Exception as exc:
        logger.error("ResumeOptimizationAgent: update_optimizations_yaml failed - %s", exc)
        return False

def log_agent_activity(action: str, status: str = "success") -> None:
    try:
        append_log_entry({
            "agent": "ResumeOptimizationAgent",
            "action": action,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except:
        pass

def run_resume_optimization_agent() -> list[dict]:
    logger.info("ResumeOptimizationAgent: Starting...")
    
    # 1. Extract resume skills
    resume_skills = []
    if download_resume_pdf():
        resume_skills = extract_skills_from_pdf()
        
    if not resume_skills:
        logger.info("ResumeOptimizationAgent: Falling back to users.yaml skills.")
        resume_skills = get_fallback_skills()
        
    if not resume_skills:
        logger.warning("ResumeOptimizationAgent: No resume skills found.")
        resume_skills = ["Python", "SQL"]
        
    # 2. Get jobs
    jobs = read_jobs()
    if not jobs:
        log_agent_activity("Skipped resume optimization - no jobs found", "partial")
        return []

    results = []
    yaml_db_entries = []
    
    resume_lower = {s.lower() for s in resume_skills}
    
    for job in jobs:
        company = job.get("company", "Unknown Company")
        role = job.get("role", "Unknown Role")
        tech_skills = job.get("technical_skills", [])
        
        job_skill_set = {str(s).strip() for s in tech_skills if str(s).strip()}
        missing = sorted({s for s in job_skill_set if s.lower() not in resume_lower})
        
        # 3. Generate suggestions
        suggestions = generate_suggestions(job, resume_skills, missing)
        
        # 4. Generate markdown and save
        github_link = save_optimized_resume_to_github(company, role, suggestions, missing)
        
        yaml_db_entries.append({
            "company": company,
            "role": role,
            "missing_skills": missing,
            "suggestions": suggestions,
            "optimized_resume_link": github_link
        })
        
        if github_link:
            results.append({
                "company": company,
                "suggestions": suggestions,
                "optimized_resume_link": github_link
            })
            
    # 5. Save YAML
    if yaml_db_entries:
        update_optimizations_yaml(yaml_db_entries)
        log_agent_activity(f"Generated optimizations for {len(yaml_db_entries)} internships")
        
    logger.info("ResumeOptimizationAgent: Completed successfully.")
    return results

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    res = run_resume_optimization_agent()
    print("Optimization results:", len(res))
