"""
per_internship_portfolio_agent.py — Per-Internship Portfolio Generator
OrchestrAI Autonomous Multi-Agent System

Generates a customized single-page portfolio for EACH internship,
highlighting the most relevant GitHub projects based on the job's
technical requirements.
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
openai_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL) if GEMINI_API_KEY else None

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
    """Score and rank projects by relevance to job's required skills."""
    job_skill_set = {s.lower() for s in job_skills}
    scored = []
    for project in projects:
        name = project.get("name", "").lower()
        desc = str(project.get("description", "")).lower()
        techs = [t.lower() for t in project.get("technologies", []) if t]
        topics = [t.lower() for t in project.get("topics", []) if t]
        
        score = 0
        matched = []
        for skill in job_skill_set:
            if skill in name or skill in desc:
                score += 3
                matched.append(skill)
            elif any(skill in t for t in techs + topics):
                score += 2
                matched.append(skill)
        
        scored.append({**project, "_relevance_score": score, "_matched_skills": matched})
    
    scored.sort(key=lambda x: x["_relevance_score"], reverse=True)
    return scored

def _generate_portfolio_html(
    company: str, role: str, job_skills: list[str],
    projects: list[dict], user_name: str, user_skills: list[str],
    skill_gaps: list[str], roadmap_steps: list[str],
    cover_letter_link: str
) -> str:
    """Generate a full-page customized portfolio HTML for one internship."""
    
    # Top 4 most relevant projects
    ranked = _rank_projects_for_job(projects, job_skills)[:4]

    projects_html = ""
    for p in ranked:
        name = p.get("name", "Project")
        desc = p.get("description", "A technical project.")
        techs = ", ".join(str(t) for t in p.get("technologies", p.get("topics", [])) if t)
        stars = p.get("stars", 0)
        url = p.get("url", "#")
        demo = p.get("demo_url", "")
        matched = p.get("_matched_skills", [])
        
        tags_html = ""
        for skill in matched[:3]:
            tags_html += f'<span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:12px;font-size:12px;margin:2px;display:inline-block">✓ {skill.title()}</span>'

        demo_btn = f'<a href="{demo}" target="_blank" style="background:#1976d2;color:white;padding:6px 12px;border-radius:5px;text-decoration:none;font-size:12px;font-weight:600">Live Demo</a>' if demo and demo != "#" else ""
        projects_html += f"""
        <div style="border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:16px;background:white;box-shadow:0 2px 6px rgba(0,0,0,0.05)">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <h3 style="margin:0;color:#1a1a2e;font-size:16px">{name}</h3>
                <span style="color:#666;font-size:12px">⭐ {stars}</span>
            </div>
            <p style="color:#555;font-size:13px;margin:8px 0">{desc}</p>
            <p style="color:#888;font-size:12px;margin-bottom:8px"><b>Tech:</b> {techs or 'Python, ML'}</p>
            <div style="margin-bottom:10px">{tags_html}</div>
            <div style="display:flex;gap:10px">
                <a href="{url}" target="_blank" style="background:#333;color:white;padding:6px 12px;border-radius:5px;text-decoration:none;font-size:12px;font-weight:600">GitHub →</a>
                {demo_btn}
            </div>
        </div>
        """

    gap_items = "".join(f"<li style='margin:4px 0;color:#d32f2f'>❌ {g}</li>" for g in (skill_gaps or ["All required skills covered!"]))
    road_items = "".join(f"<li style='margin:4px 0;color:#1565c0'>→ {r}</li>" for r in (roadmap_steps[:5] or ["Keep building projects!"]))
    skill_badges = "".join(f'<span style="background:#e3f2fd;color:#0d47a1;padding:4px 10px;border-radius:20px;font-size:12px;margin:3px;display:inline-block">{s}</span>' for s in user_skills[:10])
    cl_html = f'<a href="{cover_letter_link}" target="_blank" style="background:#2e7d32;color:white;padding:10px 20px;border-radius:6px;text-decoration:none;font-weight:600;font-size:14px">📄 View My Cover Letter →</a>' if cover_letter_link and cover_letter_link != "#" else ""
    required_skills_html = ", ".join(f"<b>{s.title()}</b>" for s in job_skills[:6])
    ts = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{user_name} — Portfolio for {role} at {company}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#f8f9fa; color:#1a1a2e; }}
  .hero {{ background:linear-gradient(135deg,#1a237e,#283593,#1565c0); color:white; padding:60px 40px; text-align:center; }}
  .hero h1 {{ font-size:32px; font-weight:700; margin-bottom:8px; }}
  .hero h2 {{ font-size:18px; font-weight:400; opacity:0.85; margin-bottom:16px; }}
  .hero .tag {{ background:rgba(255,255,255,0.2); padding:6px 16px; border-radius:20px; font-size:13px; display:inline-block; }}
  .container {{ max-width:900px; margin:0 auto; padding:40px 20px; }}
  .card {{ background:white; border-radius:12px; padding:28px; margin-bottom:24px; box-shadow:0 2px 12px rgba(0,0,0,0.07); }}
  .card h3 {{ color:#1a237e; font-size:18px; margin-bottom:16px; border-bottom:2px solid #e8eaf6; padding-bottom:10px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media(max-width:600px) {{ .grid2 {{ grid-template-columns:1fr; }} .hero {{ padding:40px 20px; }} }}
  .footer {{ text-align:center; color:#888; font-size:12px; padding:30px; }}
</style>
</head>
<body>
<div class="hero">
  <h1>{user_name}</h1>
  <h2>Customized Portfolio for <strong>{role}</strong> at <strong>{company}</strong></h2>
  <span class="tag">Generated on {ts}</span>
  {f'<br/><br/>{cl_html}' if cl_html else ''}
</div>

<div class="container">
  <div class="card">
    <h3>🎯 Why I'm a Great Fit for This Role</h3>
    <p style="color:#555;margin-bottom:12px">This role requires: {required_skills_html}</p>
    <div style="display:flex;flex-wrap:wrap;margin-top:8px">{skill_badges}</div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3>📊 Skill Gap Analysis</h3>
      <p style="font-size:12px;color:#888;margin-bottom:10px">Skills I'm actively building:</p>
      <ul style="list-style:none;padding:0">{gap_items}</ul>
    </div>
    <div class="card">
      <h3>🗺️ My Learning Roadmap</h3>
      <p style="font-size:12px;color:#888;margin-bottom:10px">My 4-week plan to close the gaps:</p>
      <ul style="list-style:none;padding:0">{road_items}</ul>
    </div>
  </div>

  <div class="card">
    <h3>🏆 Most Relevant Projects for This Role</h3>
    {projects_html or '<p style="color:#888">No projects matched yet — check back soon!</p>'}
  </div>
</div>

<div class="footer">Generated by OrchestrAI • Autonomous Career Intelligence System • {ts}</div>
</body>
</html>"""

def run_per_internship_portfolio_agent() -> list[dict]:
    logger.info("PerInternshipPortfolioAgent: Starting...")

    jobs_data = read_yaml_from_github(JOBS_FILE)
    jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []

    portfolio_data = read_yaml_from_github(PORTFOLIO_FILE)
    projects = []
    if isinstance(portfolio_data, dict):
        projects = portfolio_data.get("portfolio", {}).get("projects", [])

    users_data = read_yaml_from_github(USERS_FILE)
    user = users_data.get("user", {}) if isinstance(users_data, dict) else {}
    user_name = user.get("name", DEFAULT_USER_NAME)
    user_skills = user.get("resume_skills", DEFAULT_SKILLS)

    skill_gap_data = read_yaml_from_github("database/skill_gap_per_job.yaml")
    skill_analysis = skill_gap_data.get("job_skill_analysis", []) if isinstance(skill_gap_data, dict) else []
    skill_lookup = {(item.get("company"), item.get("role")): item for item in skill_analysis if isinstance(item, dict)}

    cl_data = read_yaml_from_github("database/cover_letter_index.yaml")
    cl_list = cl_data.get("cover_letters", []) if isinstance(cl_data, dict) else []
    cl_lookup = {(item.get("company"), item.get("role")): item.get("link", "") for item in cl_list if isinstance(item, dict)}

    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    index = []
    generated = 0

    for job in jobs[:20]:  # Cap at 20 to avoid rate limits
        company = job.get("company", "Unknown")
        role = job.get("role", "Intern")
        job_skills = [str(s) for s in job.get("technical_skills", []) if s]

        key = (company, role)
        skill_info = skill_lookup.get(key, {})
        skill_gaps = skill_info.get("missing_skills", [])
        roadmap = skill_info.get("roadmap", [])
        cl_link = cl_lookup.get(key, "")

        try:
            html = _generate_portfolio_html(
                company=company, role=role, job_skills=job_skills,
                projects=projects, user_name=user_name, user_skills=user_skills,
                skill_gaps=skill_gaps, roadmap_steps=roadmap,
                cover_letter_link=cl_link
            )
            slug = f"{_slugify(company)}_{_slugify(role)}"
            file_path = f"frontend/portfolio/internships/{slug}.html"
            _, sha = _get_raw_file(file_path)
            _put_raw_file(file_path, html, sha, f"feat: per-internship portfolio for {company} {role}")

            pub_url = f"{base_url}/portfolio/internships/{slug}.html"
            index.append({"company": company, "role": role, "portfolio_url": pub_url})
            generated += 1
            logger.info("PerInternshipPortfolioAgent: ✓ %s — %s", company, role)
        except Exception as exc:
            logger.error("PerInternshipPortfolioAgent: Failed for %s %s - %s", company, role, exc)

    write_yaml_to_github(PER_INTERNSHIP_INDEX_FILE, {"per_internship_portfolios": index})
    append_log_entry({
        "agent": "PerInternshipPortfolioAgent",
        "action": f"Generated {generated} per-internship portfolio pages",
        "status": "success" if generated > 0 else "partial",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    })

    logger.info("PerInternshipPortfolioAgent: Done. %d pages generated.", generated)
    return index


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    results = run_per_internship_portfolio_agent()
    print(f"\nGenerated {len(results)} per-internship portfolio pages.")
