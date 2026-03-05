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
    """Score and rank projects by relevance to job's required skills."""
    job_skill_set = {s.lower() for s in job_skills}
    scored = []
    for project in projects:
        # Match against title, summary, and technologies string
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
        name = p.get("title", p.get("original_name", "Project"))
        desc = p.get("summary", "A technical project.")
        techs = str(p.get("technologies", "Python, ML"))
        stars = p.get("stars", 0)  # still keep stars if present
        url = p.get("github_link", "#")
        demo = p.get("demo_link", "")
        matched = p.get("_matched_skills", [])
        
        tags_html = ""
        for skill in matched[:3]:
            tags_html += f'<span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.2);color:#10b981;padding:4px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;display:inline-block">✓ {skill.title()}</span> '

        demo_btn = f'<a href="{demo}" target="_blank" style="background:#f8fafc;color:#09090b;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:0.85rem;font-weight:600;transition:transform 0.2s">Live Demo</a>' if demo and demo != "#" else ""
        projects_html += f"""
        <div style="border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:24px;background:rgba(0,0,0,0.2);transition:border-color 0.2s;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                <h3 style="margin:0;color:#fff;font-size:1.2rem;font-weight:600;font-family:'Outfit',sans-serif;">{name}</h3>
                <span style="background:rgba(255,255,255,0.05);color:var(--text-dim);border-radius:20px;padding:4px 10px;font-size:0.75rem;">⭐ {stars}</span>
            </div>
            <p style="color:var(--text-dim);font-size:0.95rem;margin-bottom:16px;line-height:1.5;">{desc}</p>
            <p style="color:var(--primary);font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;font-weight:600;"><b>Tech:</b> {techs or 'Python, ML'}</p>
            <div style="margin-bottom:20px;display:flex;flex-wrap:wrap;gap:8px;">{tags_html}</div>
            <div style="display:flex;gap:12px;border-top:1px solid rgba(255,255,255,0.05);padding-top:16px;">
                <a href="{url}" target="_blank" style="background:transparent;border:1px solid var(--border);color:white;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:0.85rem;font-weight:600;transition:background 0.2s">GitHub &rarr;</a>
                {demo_btn}
            </div>
        </div>
        """

    gap_items = "".join(f"<li style='display:flex;gap:10px;'><span style='color:var(--error);'>x</span><span>{g}</span></li>" for g in (skill_gaps or ["All required skills covered!"]))
    road_items = "".join(f"<li style='display:flex;gap:10px;'><span style='color:var(--primary);'>&rarr;</span><span>{r}</span></li>" for r in (roadmap_steps[:5] or ["Keep building projects!"]))
    skill_badges = "".join(f'<span style="background:rgba(255,255,255,0.05);border:1px solid var(--border);color:var(--text-main);padding:6px 14px;border-radius:20px;font-size:0.85rem;font-weight:500;">{s}</span>' for s in user_skills[:10])
    required_skills_html = ", ".join(f"<b>{s.title()}</b>" for s in job_skills[:6])
    ts = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{user_name} — Portfolio for {role} at {company}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@800&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg-main: #09090b;
    --primary: #3b82f6;
    --secondary: #8b5cf6;
    --surface: rgba(255, 255, 255, 0.03);
    --border: rgba(255, 255, 255, 0.08);
    --text-main: #f8fafc;
    --text-dim: #94a3b8;
    --accent: #10b981;
    --error: #ef4444;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ 
      font-family:'Inter', sans-serif; 
      background-color: var(--bg-main); 
      color: var(--text-main); 
      background-image: radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 30%),
                        radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.08), transparent 30%);
      background-attachment: fixed;
      line-height: 1.6;
  }}
  .hero {{ 
      padding:80px 40px; 
      text-align:center;
      border-bottom: 1px solid var(--border);
      position: relative;
  }}
  .hero h1 {{ 
      font-family: 'Outfit', sans-serif;
      font-size:3rem; 
      font-weight:800; 
      margin-bottom:12px; 
      background: linear-gradient(135deg, #ffffff 30%, #a8a29e 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -1px;
  }}
  .hero h2 {{ 
      font-size:1.25rem; 
      font-weight:400; 
      color: var(--text-dim); 
      margin-bottom:24px; 
  }}
  .hero .tag {{ 
      background: rgba(59, 130, 246, 0.1); 
      color: var(--primary);
      padding:6px 16px; 
      border-radius:20px; 
      font-size:13px; 
      font-weight: 600;
      border: 1px solid rgba(59, 130, 246, 0.2);
      display:inline-block; 
      text-transform: uppercase;
      letter-spacing: 1px;
  }}
  .container {{ max-width:900px; margin:0 auto; padding:40px 20px; }}
  .card {{ 
      background: var(--surface); 
      border: 1px solid var(--border);
      border-radius:16px; 
      padding:32px; 
      margin-bottom:24px; 
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  }}
  .card h3 {{ 
      color: #fff; 
      font-family: 'Outfit', sans-serif;
      font-size:1.5rem; 
      margin-bottom:20px; 
      padding-bottom:12px; 
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 10px;
  }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom: 24px; }}
  @media(max-width:768px) {{ .grid2 {{ grid-template-columns:1fr; }} .hero {{ padding:60px 20px; }} }}
  .footer {{ text-align:center; color: var(--text-dim); font-size:13px; padding:40px; border-top: 1px solid var(--border); }}
  
  ul li {{ color: var(--text-main); font-size: 0.95rem; line-height: 1.5; }}
</style>
</head>
<body>
<div class="hero">
  <span class="tag">Generated on {ts}</span>
  <h1 style="margin-top:20px;">{user_name}</h1>
  <h2>Customized Portfolio for <strong style="color:var(--primary);">{role}</strong> at <strong style="color:var(--secondary);">{company}</strong></h2>
  {f'<br/><a href="{cover_letter_link}" target="_blank" style="display:inline-block;background:var(--primary);color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;box-shadow:0 4px 15px rgba(59,130,246,0.4);transition:all 0.2s;">📄 View Cover Letter &rarr;</a>' if cover_letter_link and cover_letter_link != "#" else ''}
</div>

<div class="container">
  <div class="card">
    <h3>🎯 Why I'm a Great Fit</h3>
    <p style="color:var(--text-dim);margin-bottom:15px;font-size:0.95rem;">This role requires: <span style="color:#e2e8f0">{required_skills_html}</span></p>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px">{skill_badges}</div>
  </div>

  <div class="grid2">
    <div class="card" style="margin-bottom:0;">
      <h3>📊 Skill Gap Analysis</h3>
      <p style="font-size:12px;color:var(--text-dim);margin-bottom:15px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Skills I'm actively building</p>
      <ul style="list-style:none;padding:0;display:flex;flex-direction:column;gap:10px;">{gap_items}</ul>
    </div>
    <div class="card" style="margin-bottom:0;">
      <h3>🗺️ My Learning Roadmap</h3>
      <p style="font-size:12px;color:var(--text-dim);margin-bottom:15px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">My action plan</p>
      <ul style="list-style:none;padding:0;display:flex;flex-direction:column;gap:10px;">{road_items}</ul>
    </div>
  </div>

  <div class="card">
    <h3>🏆 Most Relevant Projects</h3>
    <div style="display:flex;flex-direction:column;gap:16px;">
        {projects_html or '<p style="color:var(--text-dim); font-style:italic;">No mapped projects found.</p>'}
    </div>
  </div>
</div>

<div class="footer">Built autonomously by OrchestrAI • Advanced Career Intelligence System • {ts}</div>
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

    from backend.github_yaml_db import DATA_DIR
    internships_dir = os.path.join(DATA_DIR, "frontend", "portfolio", "internships")
    os.makedirs(internships_dir, exist_ok=True)

    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    index = []
    generated = 0

    for job in jobs:  # Process all fetched internships
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

            slug = f"{_slugify(company)}_{_slugify(role)}"
            # Write HTML file via GitHub Persistence layer
            file_name = f"{slug}.html"
            file_path = f"frontend/portfolio/internships/{file_name}"
            
            _, sha = _get_raw_file(file_path)
            ts = datetime.now(timezone.utc).isoformat()
            _put_raw_file(file_path, html, sha, f"feat(portfolio): generated for {company} — {ts}")

            # URL via Render's static file server
            pub_url = f"{base_url}/portfolio/internships/{slug}.html"
            index.append({"company": company, "role": role, "portfolio_url": pub_url})
            generated += 1
            logger.info("PerInternshipPortfolioAgent: ✓ %s — %s → %s", company, role, pub_url)
        except Exception as exc:
            logger.error("PerInternshipPortfolioAgent: Failed for %s %s - %s", company, role, exc)

    # Save index to GitHub so ExecutionAgent can read pre-built URLs
    try:
        write_yaml_to_github(PER_INTERNSHIP_INDEX_FILE, {"per_internship_portfolios": index})
    except Exception as exc:
        logger.error("PerInternshipPortfolioAgent: Failed to save index YAML - %s", exc)

    try:
        append_log_entry({
            "agent": "PerInternshipPortfolioAgent",
            "action": f"Generated {generated} per-internship portfolio pages",
            "status": "success" if generated > 0 else "partial",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except Exception:
        pass

    logger.info("PerInternshipPortfolioAgent: Done. %d pages generated.", generated)
    return index



if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    results = run_per_internship_portfolio_agent()
    print(f"\nGenerated {len(results)} per-internship portfolio pages.")
