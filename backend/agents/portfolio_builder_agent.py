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
) if OPENAI_API_KEY else None

USERS_FILE = "database/users.yaml"
PORTFOLIO_FILE = "database/portfolio.yaml"
SECURITY_REPORTS_FILE = "database/security_reports.yaml"

def _get_public_url(file_path: str) -> str:
    base_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:10000")
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

def _generate_project_description(repo: dict, readme: str) -> dict:
    if not openai_client:
        return {
            "title": repo.get("name", "Project"),
            "summary": repo.get("description", ""),
            "technologies": repo.get("language", "Python"),
            "impact_statement": "Contributed to open source community."
        }
    
    prompt = f"""You are a technical portfolio writer.
Rewrite the following GitHub repository into a professional project description for a portfolio website.
Include:
- Problem statement
- Solution
- Technologies used
- Key impact

Repository name: {repo.get("name")}
Description: {repo.get("description", "")}
Readme content: {readme}

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

def _generate_summary(name: str, skills: list, career_goals: list) -> str:
    if not openai_client:
        return f"I am a passionate software engineer specializing in {', '.join(skills[:3])}."
    
    prompt = f"Write a professional, impressive 2-3 sentence portfolio introduction for a software engineer named {name} with skills in {skills} and career goals {career_goals}."
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip().replace('"', '')
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
            <h3>{p.get('title')}</h3>
            <p class="tech">{p.get('technologies')}</p>
            <p>{p.get('summary')}</p>
            <p class="impact"><em>{p.get('impact_statement')}</em></p>
            {sec_html}
            <div class="card-footer">
                <a href="{gh_url}" class="gh-btn" target="_blank">GitHub</a>
                {demo_btn}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} - Portfolio</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #cbd5e1;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            text-align: center;
            padding: 4rem 1rem;
            border-bottom: 1px solid #334155;
        }}
        h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .summary {{
            font-size: 1.25rem;
            color: var(--text-muted);
            max-width: 700px;
            margin: 0 auto 2rem auto;
        }}
        .skills {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.5rem;
        }}
        .skill-badge {{
            background: #334155;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 500;
        }}
        section {{
            padding: 4rem 0;
        }}
        section h2 {{
            font-size: 2rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--accent);
            display: inline-block;
            padding-bottom: 0.5rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        .project-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
        }}
        .project-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        }}
        .project-card h3 {{ margin-top: 0; margin-bottom: 0.5rem; }}
        .project-card .tech {{
            color: #60a5fa;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }}
        .project-card .impact {{
            font-size: 0.9rem;
            color: #a78bfa;
            margin-top: auto;
            padding-top: 1rem;
        }}
        .card-footer {{
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            align-items: center;
        }}
        .gh-btn, .demo-btn {{
            padding: 0.5rem 1rem;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .gh-btn {{
            background: #334155;
            color: white;
            border: 1px solid #475569;
        }}
        .gh-btn:hover {{ background: #475569; }}
        .demo-btn {{
            background: var(--accent);
            color: white;
        }}
        .demo-btn:hover {{ background: var(--accent-hover); }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            border-top: 1px solid #334155;
            margin-top: 4rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{name}</h1>
            <p class="summary">{summary}</p>
            <div class="skills">
                {skills_html}
            </div>
        </div>
    </header>

    <div class="container">
        <section id="projects">
            <h2>Featured Projects</h2>
            <div class="grid">
                {projects_html}
            </div>
        </section>
    </div>

    <footer>
        <p>Built with OrchestrAI</p>
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
    summary = _generate_summary(user.get("name", ""), skills, career_goals)
    
    # 3. Generate Project info
    project_list = []
    for repo in repos:
        name = repo.get("name")
        readme = _get_readme(github_username, name)
        desc = _generate_project_description(repo, readme)
        
        has_demo = any(topic in repo.get("topics", []) for topic in ["streamlit", "fastapi", "flask"]) or \
                   any(tech in desc.get("technologies", "").lower() for tech in ["streamlit", "fastapi", "flask"])
        
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
