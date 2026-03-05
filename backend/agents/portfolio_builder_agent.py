"""
portfolio_builder_agent.py — Portfolio Builder Agent
OrchestrAI Autonomous Multi-Agent System

Automatically generate a professional personal portfolio website by curating GitHub repositories, 
generating project descriptions using an LLM, and deploying the portfolio through the existing Render deployment pipeline.
"""

from __future__ import annotations

import logging
import os
import requests
from datetime import datetime, timezone
import json

from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
    DATA_DIR
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.PortfolioBuilderAgent")

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=GEMINI_BASE_URL,
    max_retries=0,
) if OPENAI_API_KEY else None

from backend.utils.ai_engine import safe_llm_call as _cb_llm_call
from backend.utils.resume_parser import download_and_extract

USERS_FILE = "database/users.yaml"
PORTFOLIO_FILE = "database/portfolio.yaml"
SECURITY_REPORTS_FILE = "database/security_reports.yaml"

def _get_public_url(file_path: str) -> str:
    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    # For a directory mount with html=True in FastAPI, it works nicely to point directly to the HTML file or the enclosing dir.
    # To keep it completely in sync with the user spec: /portfolio
    return f"{base_url}/portfolio/"

def fetch_github_repos(username: str) -> list[dict]:
    headers = {}
    github_token = os.getenv("GITHUB_TOKEN", "")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            repos = resp.json()
            valid_repos = []
            for repo in repos:
                if repo.get("fork") or repo.get("archived") or not repo.get("size", 0):
                    continue
                valid_repos.append(repo)
            # Sort by stars and updated
            valid_repos.sort(key=lambda x: (x.get("stargazers_count", 0), x.get("updated_at", "")), reverse=True)
            return valid_repos[:8]  # top 8
    except Exception as exc:
        logger.error("PortfolioBuilderAgent: Failed to fetch repos - %s", exc)
    return []

def _get_readme(username: str, repo: str) -> str:
    headers = {}
    github_token = os.getenv("GITHUB_TOKEN", "")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    url = f"https://raw.githubusercontent.com/{username}/{repo}/main/README.md"
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.text[:1000] # Provide excerpt
        url_master = f"https://raw.githubusercontent.com/{username}/{repo}/master/README.md"
        resp = requests.get(url_master, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.text[:1000]
    except:
        pass
    return ""

def _generate_project_description(repo: dict, readme: str, resume_text: str = "") -> dict:
    if not openai_client:
        return {
            "title": repo.get("name", "Project"),
            "summary": repo.get("description", ""),
            "technologies": repo.get("language", "Python"),
            "impact_statement": "Contributed to open source community."
        }
    
    prompt = f"""You are a technical portfolio writer.
Rewrite the following GitHub repository into a professional project description for a portfolio website.
I will also provide the user's resume. If the repository relates to any projects mentioned in the resume, incorporate those details (like actual impact numbers, specific tools used) into the description to make it highly personalized and accurate.

Include:
- Problem statement
- Solution
- Technologies used
- Key impact

Repository name: {repo.get("name")}
Description: {repo.get("description", "")}
Readme content: {readme}

User Resume Text:
{resume_text[:2000]}

Return exactly in valid JSON format:
{{
  "title": "Clean readable title based on repo name",
  "summary": "2-3 sentences max combining problem and solution",
  "technologies": "comma separated tech stack",
  "impact_statement": "1 sentence key impact"
}}
"""
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data
    except Exception as exc:
        logger.warning(f"PortfolioBuilderAgent: LLM generated failed for {repo.get('name')} - {exc}")
        return {
            "title": repo.get("name", "Project"),
            "summary": str(repo.get("description", ""))[:150],
            "technologies": repo.get("language", "Python"),
            "impact_statement": "No impact metric formulated."
        }

def _generate_summary(name: str, skills: list, career_goals: list, resume_text: str = "") -> str:
    if not openai_client:
        return f"I am a passionate software engineer specializing in {', '.join(skills[:3])}."
    
    prompt = f"Write a professional, highly impressive 2-3 sentence portfolio introduction for a software engineer named {name}. Their technical skills include {skills} and their career goals are {career_goals}. Use details from their resume to make this summary truly personalized to their actual experience: \n\nResume excerpt:\n{resume_text[:2000]}"
    try:
        response_text = _cb_llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            context="portfolio_summary"
        )
        if response_text:
            return response_text.strip().replace('"', '')
    except Exception:
        return f"I am a passionate software engineer specializing in {', '.join(skills[:3])}."

def _render_portfolio_html(user: dict, summary: str, projects: list[dict], security_reports: list[dict]) -> str:
    name = user.get("name", "Applicant")
    skills = user.get("skills", user.get("resume_skills", ["Python", "Machine Learning", "FastAPI"]))
    
    skills_html = "".join([f'<span class="skill-badge">{s}</span>' for s in skills])
    
    sec_lookup = {
        item.get("repo", ""): item
        for item in security_reports if isinstance(item, dict)
    }

    projects_html = ""
    for p in projects:
        gh_url = p.get("github_link", "#")
        demo_url = p.get("demo_link", gh_url)
        orig_name = p.get("original_name", "")
        
        if p.get("demo_link") != "#":
            demo_btn = f'<a href="{demo_url}" class="demo-btn" target="_blank">Live Demo</a>'
        else:
            demo_btn = f'<img src="https://img.shields.io/github/stars/{user.get("github_username")}/{orig_name}?style=social" alt="GitHub stars" style="margin-right:10px;">'

        sec_data = sec_lookup.get(orig_name, {})
        sec_score = sec_data.get("risk_score", 0)
        sec_issues = sec_data.get("issues", [])
        
        if sec_data:
            badge_color = "#2e7d32" if sec_score <= 1 else "#f29900" if sec_score <= 4 else "#d32f2f"
            issues_snippet = "<br>".join([f"&bull; {str(issue).replace('<', '&lt;').replace('>', '&gt;')}" for issue in sec_issues[:2]])
            sec_html = f"""
            <div style="margin-top:0.5rem; padding:0.5rem; background:rgba(0,0,0,0.2); border-radius:6px; font-size:0.8rem;">
                <span style="color:{badge_color}; font-weight:bold;">Security Risk: {sec_score}</span>
                <div style="margin-top:0.2rem; color:var(--text-muted);">{issues_snippet}</div>
            </div>
            """
        else:
            sec_html = ""

        projects_html += f"""
        <div class="project-card">
            <h3>{p.get("title", "")}</h3>
            <p class="tech">{p.get("technologies", "")}</p>
            <p>{p.get("summary", "")}</p>
            <div class="impact"><em>" {p.get("impact_statement", "")} "</em></div>
            {sec_html}
            <div class="card-footer">
                <a href="{gh_url}" class="btn gh-btn" target="_blank">GitHub Repo</a>
                {demo_btn.replace('demo-btn', 'btn demo-btn')}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} — AI & Data Portfolio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #09090b;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.5);
            --secondary: #8b5cf6;
            --surface: rgba(255, 255, 255, 0.03);
            --surface-hover: rgba(255, 255, 255, 0.08);
            --border: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%);
            background-attachment: fixed;
            line-height: 1.7;
        }}
        
        .container {{ max-width: 1100px; margin: 0 auto; padding: 0 2rem; }}
        
        /* Hero Section */
        header {{
            padding: 8rem 0 5rem 0;
            text-align: center;
            border-bottom: 1px solid var(--border);
            position: relative;
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(59, 130, 246, 0.1);
            color: var(--primary);
            padding: 0.5rem 1rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 1px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            text-transform: uppercase;
        }}
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #ffffff 30%, #a8a29e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }}
        .summary {{
            font-size: 1.25rem;
            color: var(--text-dim);
            max-width: 700px;
            margin: 0 auto 3rem auto;
            font-weight: 300;
        }}
        .skills {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.75rem;
            max-width: 800px;
            margin: 0 auto;
        }}
        .skill-badge {{
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 0.6rem 1.25rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        .skill-badge:hover {{
            background: var(--surface-hover);
            border-color: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }}

        /* Projects Section */
        section {{ padding: 6rem 0; }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 4rem;
        }}
        .section-header h2 {{
            font-size: 2.25rem;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
        }}
        .section-line {{
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--border), transparent);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 2rem;
        }}
        
        .project-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }}
        .project-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .project-card:hover {{
            transform: translateY(-8px);
            border-color: rgba(255,255,255,0.15);
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
        }}
        .project-card:hover::before {{ opacity: 1; }}

        .project-card h3 {{
            font-size: 1.4rem;
            margin-top: 0;
            margin-bottom: 0.5rem;
            color: #fff;
        }}
        .project-card .tech {{
            color: var(--primary);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1.25rem;
            font-weight: 700;
        }}
        .project-card p {{
            color: var(--text-dim);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }}
        .project-card .impact {{
            color: #e2e8f0;
            margin-top: auto;
            border-left: 3px solid var(--secondary);
            padding-left: 1rem;
            font-style: italic;
            background: rgba(139, 92, 246, 0.05);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
        }}

        .card-footer {{
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }}
        .btn {{
            padding: 0.6rem 1.25rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            cursor: pointer;
        }}
        .gh-btn {{
            background: transparent;
            color: #fff;
            border: 1px solid var(--border);
        }}
        .gh-btn:hover {{
            background: rgba(255,255,255,0.05);
            border-color: rgba(255,255,255,0.2);
        }}
        .demo-btn {{
            background: var(--text-main);
            color: var(--bg-dark);
            border: 1px solid transparent;
        }}
        .demo-btn:hover {{
            background: #d1d5db;
            transform: scale(1.02);
        }}

        footer {{
            text-align: center;
            padding: 3rem;
            color: var(--text-dim);
            border-top: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            h1 {{ font-size: 3rem; }}
            header {{ padding: 5rem 0 3rem 0; }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="hero-badge">Curated Engineering Portfolio</div>
            <h1>{name}</h1>
            <p class="summary">{summary}</p>
            <div class="skills">
                {skills_html}
            </div>
        </div>
    </header>

    <div class="container">
        <section id="projects">
            <div class="section-header">
                <h2>Featured Projects</h2>
                <div class="section-line"></div>
            </div>
            <div class="grid">
                {projects_html}
            </div>
        </section>
    </div>

    <footer>
        <p>Built autonomously by OrchestrAI</p>
    </footer>
</body>
</html>"""
    return html

def save_portfolio_page(html: str) -> str:
    file_path = "frontend/portfolio/index.html"
    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(file_path, html, sha, f"feat: generated portfolio - {ts}")
        return _get_public_url("frontend/portfolio/index.html")
    except Exception as exc:
        logger.error("PortfolioBuilderAgent: Failed to save portfolio - %s", exc)
        return ""

def log_agent_activity(status: str) -> None:
    try:
        append_log_entry({
            "agent": "PortfolioBuilderAgent",
            "action": "Generated Portfolio",
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except:
        pass

def save_portfolio_yaml(metadata: dict) -> bool:
    try:
        return write_yaml_to_github(PORTFOLIO_FILE, {"portfolio": metadata})
    except Exception as exc:
        logger.error("PortfolioBuilderAgent: Failed to save yam - %s", exc)
        return False

def run_portfolio_builder_agent():
    logger.info("PortfolioBuilderAgent: Starting...")
    
    data = {"user": {}}
    try:
        data = read_yaml_from_github(USERS_FILE)
    except:
        logger.warning("PortfolioBuilderAgent: Could not read users.yaml")
        
    user = data.get("user", {})
    if not user:
        # Fallback to defaults
        user = {
            "name": "Applicant",
            "github_username": "Swathy1209",
            "skills": ["Python", "Machine Learning"],
            "career_goals": ["AI Engineer"]
        }
        
    github_username = user.get("github_username", "Swathy1209")
    if not github_username:
        logger.warning("PortfolioBuilderAgent: No GitHub username found.")
        return None
        
    # 1. Fetch Repos
    repos = fetch_github_repos(github_username)
    if not repos:
        logger.warning("PortfolioBuilderAgent: No valid repos found.")
        return None
        
    # 2. Generate Summary
    skills = user.get("skills", user.get("resume_skills", ["Python", "Machine Learning"]))
    career_goals = user.get("career_goals", ["AI Engineer"])
    
    resume_text = ""
    try:
        resume_text = download_and_extract()
    except Exception as exc:
        logger.warning(f"PortfolioBuilderAgent: Failed to fetch resume - {exc}")
        
    summary = _generate_summary(user.get("name", ""), skills, career_goals, resume_text)
    
    # 3. Generate Project info
    project_list = []
    for repo in repos:
        name = repo.get("name")
        readme = _get_readme(github_username, name)
        desc = _generate_project_description(repo, readme, resume_text)
        
        has_demo = any(topic in repo.get("topics", []) for topic in ["streamlit", "fastapi", "flask"]) or \
                   any(tech in str(desc.get("technologies") or "").lower() for tech in ["streamlit", "fastapi", "flask"])
        
        demo_link = f"https://{name}.onrender.com" if has_demo else "#"
        
        project_list.append({
            "original_name": name,
            "title": desc.get("title", name),
            "summary": desc.get("summary", ""),
            "technologies": desc.get("technologies", ""),
            "impact_statement": desc.get("impact_statement", ""),
            "github_link": repo.get("html_url", ""),
            "demo_link": demo_link
        })
        
    # 3.5. Fetch Security Reports
    sec_data_obj = {}
    try:
        sec_data_obj = read_yaml_from_github(SECURITY_REPORTS_FILE)
    except:
        pass
    security_reports = sec_data_obj.get("security_reports", [])
        
    # 4. Render HTML
    html = _render_portfolio_html(user, summary, project_list, security_reports)
    url = save_portfolio_page(html)
    
    # 5. Metadata
    metadata = {
        "url": url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "projects": project_list
    }
    save_portfolio_yaml(metadata)
    log_agent_activity("success")
    logger.info(f"PortfolioBuilderAgent: Generated portfolio at {url}")
    return url

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    run_portfolio_builder_agent()
