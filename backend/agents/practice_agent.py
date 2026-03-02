"""
practice_agent.py — Interview Practice & Coaching Agent
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Read database/jobs.yaml         (internships)
  2. Read database/users.yaml        (user profile)
  3. Read database/skill_gap_per_job.yaml  (skill gaps per job)
  4. Download & extract resume.pdf text
  5. For each internship:
     a. Generate interview questions & answers          (OpenAI)
     b. Generate HR introduction                        (OpenAI)
     c. Generate Tamil → English interview translator   (OpenAI)
     d. Generate English speaking practice sentences    (OpenAI)
     e. Generate coding practice sheet links
     f. Generate project recommendations from skill gaps
     g. Generate course recommendations                 (OpenAI)
     h. Render & upload practice HTML page to GitHub
  6. Save database/practice_sessions.yaml
  7. Log to database/agent_logs.yaml
  8. Return practice_sessions list to ExecutionAgent
"""

from __future__ import annotations

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
    _put_raw_file,
    GITHUB_USERNAME,
    GITHUB_REPO,
)
from backend.utils.resume_parser import download_and_extract

load_dotenv()
logger = logging.getLogger("OrchestrAI.PracticeAgent")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

JOBS_FILE = "database/jobs.yaml"
USERS_FILE = "database/users.yaml"
SKILL_GAP_FILE = "database/skill_gap_per_job.yaml"
PRACTICE_SESSIONS_FILE = "database/practice_sessions.yaml"

_REPO_SLUG = GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"


# ==============================================================================
# Helper utilities
# ==============================================================================

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def _get_github_url(file_path: str) -> str:
    """Use raw.githack.com which serves files with correct Content-Type headers."""
    return f"https://raw.githack.com/{_REPO_SLUG}/main/{file_path}"


def _ai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Send a chat completion request to OpenAI. Returns empty string on failure."""
    if not openai_client:
        logger.warning("PracticeAgent: OpenAI API key missing — returning fallback.")
        return ""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("PracticeAgent: OpenAI call failed — %s", exc)
        return ""


# ==============================================================================
# Data loaders
# ==============================================================================

def read_jobs() -> list[dict]:
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    except Exception as exc:
        logger.error("PracticeAgent: read_jobs failed — %s", exc)
        return []


def read_user_profile() -> dict:
    try:
        data = read_yaml_from_github(USERS_FILE)
        return data.get("user", {}) if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("PracticeAgent: read_user_profile failed — %s", exc)
        return {}


def read_skill_gaps() -> list[dict]:
    try:
        data = read_yaml_from_github(SKILL_GAP_FILE)
        gaps = data.get("job_skill_analysis", [])
        return gaps if isinstance(gaps, list) else []
    except Exception as exc:
        logger.error("PracticeAgent: read_skill_gaps failed — %s", exc)
        return []


def load_resume_text() -> str:
    try:
        text = download_and_extract()
        return text
    except Exception as exc:
        logger.error("PracticeAgent: load_resume_text failed — %s", exc)
        return ""


# ==============================================================================
# FEATURE 1 — Interview Questions & Answers
# ==============================================================================

def generate_interview_qa(company: str, role: str, skills: list[str],
                          resume_text: str, user_skills: list[str]) -> list[dict]:
    """Generate top 10 interview questions + personalized answers."""
    system = (
        "You are an expert technical interviewer. Generate exactly 10 interview "
        "questions specific to the role given. For each question, produce a "
        "personalized answer using the candidate's skills and projects. "
        "Return as numbered Q&A pairs."
    )
    prompt = (
        f"Company: {company}\nRole: {role}\n"
        f"Required Skills: {', '.join(skills)}\n"
        f"Candidate Skills: {', '.join(user_skills)}\n"
        f"Resume Excerpt:\n{resume_text[:1500]}\n\n"
        "Generate 10 Q&A pairs. Format:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ..."
    )
    raw = _ai_chat(system, prompt, max_tokens=1500)
    if not raw:
        # Fallback
        return [
            {"question": f"Tell me about your experience with {s}.",
             "answer": f"I have worked with {s} in multiple projects and am eager to deepen my expertise."}
            for s in (skills[:10] if skills else ["this role"])
        ]
    # Parse Q&A
    pairs = []
    lines = raw.split("\n")
    q, a = "", ""
    for line in lines:
        line = line.strip()
        if re.match(r'^Q\d+[:.]', line):
            if q and a:
                pairs.append({"question": q, "answer": a})
            q = re.sub(r'^Q\d+[:.\s]*', '', line).strip()
            a = ""
        elif re.match(r'^A\d+[:.]', line):
            a = re.sub(r'^A\d+[:.\s]*', '', line).strip()
        elif a:
            a += " " + line
        elif q:
            q += " " + line
    if q and a:
        pairs.append({"question": q, "answer": a})
    return pairs if pairs else [{"question": "Describe your background.", "answer": "I am an aspiring professional."}]


# ==============================================================================
# FEATURE 2 — HR Introduction Generator
# ==============================================================================

def generate_hr_introduction(user: dict, company: str, role: str, resume_text: str) -> str:
    name = user.get("name", "Applicant")
    skills = ", ".join(user.get("resume_skills", []))
    goals = ", ".join(user.get("career_goals", []))

    system = (
        "You are a career coach. Generate a polished, professional self-introduction "
        "for a job interview. It must feel natural and confident."
    )
    prompt = (
        f"Candidate: {name}\nSkills: {skills}\nCareer Goals: {goals}\n"
        f"Applying for: {role} at {company}\n"
        f"Resume Excerpt:\n{resume_text[:800]}\n\n"
        "Generate a 100-word self-introduction suitable for HR round."
    )
    result = _ai_chat(system, prompt, max_tokens=300)
    if not result:
        result = (
            f"Hello, my name is {name}. I am a passionate student with experience in "
            f"{skills}. My career goals include {goals}. I am excited to apply for the "
            f"{role} position at {company} and contribute to the team."
        )
    return result


# ==============================================================================
# FEATURE 3 — Tamil → English Interview Translator
# ==============================================================================

def translate_tamil_to_interview_english(user_input: str) -> dict:
    """
    Convert Tamil text into professional interview English + simplified practice version.
    Returns dict with 'professional' and 'practice' keys.
    """
    system = (
        "You are a bilingual Tamil-English language coach specialising in "
        "interview preparation. Convert the following Tamil text into two versions: "
        "1) Professional English suitable for a job interview "
        "2) Simplified practice version for beginners. "
        "Format:\nProfessional English:\n...\nPractice Version:\n..."
    )
    raw = _ai_chat(system, user_input, max_tokens=400)
    if not raw:
        return {
            "professional": "Translation unavailable — OpenAI key not configured.",
            "practice": "Translation unavailable."
        }
    prof, prac = "", ""
    mode = None
    for line in raw.split("\n"):
        stripped = line.strip()
        if "professional" in stripped.lower() and "english" in stripped.lower():
            mode = "prof"
            continue
        if "practice" in stripped.lower() and "version" in stripped.lower():
            mode = "prac"
            continue
        if mode == "prof":
            prof += stripped + " "
        elif mode == "prac":
            prac += stripped + " "
    return {
        "professional": prof.strip() or raw,
        "practice": prac.strip() or raw,
    }


def _generate_sample_translations() -> list[dict]:
    """Return a few pre-built Tamil → English examples for the HTML page."""
    samples = [
        {
            "tamil": "nan enna intro kudukurathu",
            "professional": "Hello, allow me to introduce myself. My name is Swathiga and I am an aspiring data professional.",
            "practice": "My name is Swathiga. I am a data science student.",
        },
        {
            "tamil": "enakku Python theriyum",
            "professional": "I possess strong proficiency in Python programming and have applied it extensively in data analysis and machine learning projects.",
            "practice": "I know Python. I use it for projects.",
        },
        {
            "tamil": "nan oru project panniruken",
            "professional": "I have successfully completed a project that demonstrates my technical capabilities and problem-solving skills.",
            "practice": "I did a project. It was about building an AI model.",
        },
    ]
    return samples


# ==============================================================================
# FEATURE 4 — English Speaking Practice
# ==============================================================================

def generate_speaking_practice(role: str, skills: list[str]) -> list[dict]:
    system = (
        "You are an English speaking coach for job interviews. "
        "Generate 8 practice sentences a candidate should rehearse. "
        "Also provide 5 confidence tips. "
        "Format:\nPractice Sentences:\n1. ...\n...\nConfidence Tips:\n1. ..."
    )
    prompt = f"Role: {role}\nKey Skills: {', '.join(skills)}"
    raw = _ai_chat(system, prompt, max_tokens=600)

    sentences = []
    tips = []
    if raw:
        mode = None
        for line in raw.split("\n"):
            line = line.strip()
            if "practice" in line.lower() and "sentence" in line.lower():
                mode = "sentences"
                continue
            if "confidence" in line.lower() and "tip" in line.lower():
                mode = "tips"
                continue
            cleaned = re.sub(r'^\d+[.)]\s*', '', line).strip()
            if cleaned:
                if mode == "sentences":
                    sentences.append(cleaned)
                elif mode == "tips":
                    tips.append(cleaned)

    if not sentences:
        sentences = [
            f"I am passionate about building scalable {role.lower()} solutions.",
            "I have hands-on experience with Python, SQL, and data analysis.",
            "I thrive in collaborative environments and enjoy solving complex problems.",
            "My strongest skill is my ability to learn new technologies quickly.",
            "I am eager to apply my academic knowledge in a real-world setting.",
        ]
    if not tips:
        tips = [
            "Speak slowly and clearly.",
            "Maintain eye contact with the interviewer.",
            "Use the STAR method for behavioral questions.",
            "Take a breath before answering complex questions.",
            "Practice in front of a mirror daily.",
        ]
    return [{"sentences": sentences, "tips": tips}]


# ==============================================================================
# FEATURE 5 — Coding Practice Sheet Links
# ==============================================================================

def generate_coding_sheets(role: str, skills: list[str]) -> list[dict]:
    role_lower = role.lower()
    skills_lower = [s.lower() for s in skills]

    sheets = []

    # Universal
    sheets.append({"name": "NeetCode — DSA Practice", "url": "https://neetcode.io"})
    sheets.append({"name": "LeetCode — Problem Solving", "url": "https://leetcode.com"})

    # Data / SQL
    if any(k in role_lower for k in ("data", "analyst", "engineer")) or "sql" in skills_lower:
        sheets.append({"name": "SQLBolt — Interactive SQL", "url": "https://sqlbolt.com"})
        sheets.append({"name": "StrataScratch — SQL Interview", "url": "https://stratascratch.com"})

    # ML / AI
    if any(k in role_lower for k in ("machine learning", "ml", "ai", "deep learning")) or \
       any(s in skills_lower for s in ("pytorch", "tensorflow", "keras", "machine learning")):
        sheets.append({"name": "Kaggle — ML Competitions", "url": "https://kaggle.com"})
        sheets.append({"name": "Deep-ML — Deep Learning Practice", "url": "https://deep-ml.com"})

    # Web / Backend
    if any(k in role_lower for k in ("backend", "full stack", "web")):
        sheets.append({"name": "FreeCodeCamp — Full Stack", "url": "https://freecodecamp.org"})

    # Cloud
    if any(s in skills_lower for s in ("aws", "gcp", "azure", "cloud")):
        sheets.append({"name": "AWS Skill Builder", "url": "https://explore.skillbuilder.aws"})

    return sheets


# ==============================================================================
# FEATURE 6 — Project Recommendations
# ==============================================================================

def generate_project_recommendations(missing_skills: list[str], role: str) -> list[dict]:
    if not missing_skills:
        return [{"skill": "General", "project": f"Build a portfolio project relevant to {role}."}]

    system = (
        "You are a technical mentor. For each missing skill, suggest one concrete "
        "project idea that a student can build to learn that skill. "
        "Format: Skill: ...\nProject: ..."
    )
    prompt = f"Role: {role}\nMissing Skills: {', '.join(missing_skills)}"
    raw = _ai_chat(system, prompt, max_tokens=600)

    recommendations = []
    if raw:
        current_skill = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.lower().startswith("skill:"):
                current_skill = line.split(":", 1)[1].strip()
            elif line.lower().startswith("project:") and current_skill:
                proj = line.split(":", 1)[1].strip()
                recommendations.append({"skill": current_skill, "project": proj})
                current_skill = ""

    if not recommendations:
        for skill in missing_skills:
            recommendations.append({
                "skill": skill,
                "project": f"Build a hands-on project using {skill} (e.g., data pipeline, ML model, or API).",
            })

    return recommendations


# ==============================================================================
# FEATURE 7 — Course Recommendations
# ==============================================================================

def generate_course_recommendations(missing_skills: list[str], role: str) -> list[dict]:
    system = (
        "You are a learning advisor. For each skill, recommend 2-3 free courses or resources. "
        "Include YouTube, Coursera, and official docs where possible. "
        "Format:\nSkill: ...\n- Course: ... | Link: ..."
    )
    prompt = f"Role: {role}\nSkills to learn: {', '.join(missing_skills)}"
    raw = _ai_chat(system, prompt, max_tokens=800)

    courses = []
    if raw:
        current_skill = ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.lower().startswith("skill:"):
                current_skill = line.split(":", 1)[1].strip()
            elif line.startswith("-") and current_skill:
                courses.append({
                    "skill": current_skill,
                    "recommendation": line.lstrip("- ").strip(),
                })

    if not courses:
        for skill in missing_skills:
            courses.extend([
                {"skill": skill, "recommendation": f"YouTube: Search '{skill} full course for beginners'"},
                {"skill": skill, "recommendation": f"Coursera: {skill} Specialization"},
                {"skill": skill, "recommendation": f"Official Docs: {skill} documentation"},
            ])

    return courses


# ==============================================================================
# FEATURE 8 — Generate Practice HTML Portal
# ==============================================================================

def _render_practice_html(
    company: str,
    role: str,
    qa_pairs: list[dict],
    hr_intro: str,
    translations: list[dict],
    speaking: list[dict],
    coding_sheets: list[dict],
    projects: list[dict],
    courses: list[dict],
) -> str:
    """Render a beautiful, self-contained HTML practice portal."""

    # --- Q&A Section ---
    qa_html = ""
    for i, pair in enumerate(qa_pairs, 1):
        qa_html += f"""
        <div class="card qa-card">
            <div class="qa-q"><span class="badge">Q{i}</span> {pair['question']}</div>
            <div class="qa-a"><strong>Answer:</strong> {pair['answer']}</div>
        </div>"""

    # --- Translation Section ---
    trans_html = ""
    for t in translations:
        trans_html += f"""
        <div class="card trans-card">
            <div class="tamil"><strong>🗣 Tamil:</strong> <em>"{t['tamil']}"</em></div>
            <div class="prof"><strong>💼 Professional English:</strong> {t['professional']}</div>
            <div class="prac"><strong>📝 Practice Version:</strong> {t['practice']}</div>
        </div>"""

    # --- Speaking Practice ---
    sp_data = speaking[0] if speaking else {"sentences": [], "tips": []}
    sentences_html = "".join(f'<li class="speak-item">{s}</li>' for s in sp_data.get("sentences", []))
    tips_html = "".join(f'<li class="tip-item">{t}</li>' for t in sp_data.get("tips", []))

    # --- Coding Sheets ---
    sheets_html = ""
    for cs in coding_sheets:
        sheets_html += f"""
        <a href="{cs['url']}" target="_blank" class="sheet-link">
            <span class="sheet-icon">💻</span>
            <span>{cs['name']}</span>
            <span class="sheet-arrow">→</span>
        </a>"""

    # --- Projects ---
    projects_html = ""
    for p in projects:
        projects_html += f"""
        <div class="card project-card">
            <div class="project-skill">🔧 Missing Skill: <strong>{p['skill']}</strong></div>
            <div class="project-idea">💡 {p['project']}</div>
        </div>"""

    # --- Courses ---
    courses_html = ""
    current_skill = ""
    for c in courses:
        if c["skill"] != current_skill:
            if current_skill:
                courses_html += "</div>"
            current_skill = c["skill"]
            courses_html += f'<div class="course-group"><h4 class="course-skill">📘 {current_skill}</h4>'
        courses_html += f'<div class="course-item">• {c["recommendation"]}</div>'
    if current_skill:
        courses_html += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Practice Portal — {company} | {role}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 0;
        }}
        .hero {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .hero::after {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
            animation: pulse 8s ease-in-out infinite;
        }}
        @keyframes pulse {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
        .hero h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            color: #fff;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .hero .subtitle {{
            font-size: 1.1rem;
            color: rgba(255,255,255,0.85);
            margin-top: 10px;
            position: relative;
            z-index: 1;
        }}
        .container {{ max-width: 960px; margin: 0 auto; padding: 30px 20px; }}
        .section {{
            margin-bottom: 40px;
            animation: fadeIn 0.6s ease-out;
        }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #a78bfa;
            margin-bottom: 18px;
            border-left: 4px solid #a78bfa;
            padding-left: 14px;
        }}
        .card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 14px;
            backdrop-filter: blur(12px);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(103,126,234,0.12); }}
        .badge {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 8px;
        }}
        .qa-q {{ font-weight: 600; color: #c4b5fd; margin-bottom: 10px; font-size: 1.05rem; }}
        .qa-a {{ color: #d1d5db; line-height: 1.6; }}
        .intro-box {{
            background: linear-gradient(135deg, rgba(103,126,234,0.15), rgba(118,75,162,0.15));
            border: 1px solid rgba(167,139,250,0.25);
            border-radius: 16px;
            padding: 28px;
            font-size: 1.05rem;
            line-height: 1.7;
            color: #e2e8f0;
        }}
        .trans-card .tamil {{ color: #fbbf24; margin-bottom: 8px; }}
        .trans-card .prof {{ color: #86efac; margin-bottom: 6px; }}
        .trans-card .prac {{ color: #93c5fd; }}
        .speak-item, .tip-item {{ padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        ul {{ list-style: none; padding-left: 0; }}
        .sheet-link {{
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 10px;
            color: #e0e0e0;
            text-decoration: none;
            transition: all 0.25s;
        }}
        .sheet-link:hover {{ background: rgba(103,126,234,0.15); border-color: #667eea; transform: translateX(4px); }}
        .sheet-icon {{ font-size: 1.3rem; }}
        .sheet-arrow {{ margin-left: auto; color: #667eea; font-weight: 700; }}
        .project-skill {{ color: #fbbf24; margin-bottom: 6px; }}
        .project-idea {{ color: #d1d5db; line-height: 1.6; }}
        .course-group {{ margin-bottom: 18px; }}
        .course-skill {{ color: #86efac; margin-bottom: 8px; }}
        .course-item {{ color: #d1d5db; padding: 4px 0; padding-left: 12px; }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: rgba(255,255,255,0.35);
            font-size: 0.85rem;
        }}
        @media (max-width: 640px) {{
            .hero h1 {{ font-size: 1.6rem; }}
            .container {{ padding: 20px 14px; }}
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>🎯 Interview Practice Portal</h1>
        <p class="subtitle">{company} — {role}</p>
    </div>

    <div class="container">
        <!-- Section 1: Interview Questions & Answers -->
        <div class="section">
            <h2 class="section-title">📋 Interview Questions & Answers</h2>
            {qa_html}
        </div>

        <!-- Section 2: HR Introduction -->
        <div class="section">
            <h2 class="section-title">🎤 HR Introduction</h2>
            <div class="intro-box">{hr_intro}</div>
        </div>

        <!-- Section 3: Tamil → English Translator -->
        <div class="section">
            <h2 class="section-title">🌐 Tamil → English Interview Translator</h2>
            {trans_html}
        </div>

        <!-- Section 4: English Speaking Practice -->
        <div class="section">
            <h2 class="section-title">🗣 English Speaking Practice</h2>
            <div class="card">
                <h3 style="color:#c4b5fd; margin-bottom:12px;">Practice Sentences</h3>
                <ul>{sentences_html}</ul>
            </div>
            <div class="card">
                <h3 style="color:#86efac; margin-bottom:12px;">Confidence Tips</h3>
                <ul>{tips_html}</ul>
            </div>
        </div>

        <!-- Section 5: Coding Practice Sheets -->
        <div class="section">
            <h2 class="section-title">💻 Coding Practice Sheets</h2>
            {sheets_html}
        </div>

        <!-- Section 6: Project Recommendations -->
        <div class="section">
            <h2 class="section-title">🚀 Project Recommendations</h2>
            {projects_html}
        </div>

        <!-- Section 7: Course Recommendations -->
        <div class="section">
            <h2 class="section-title">📚 Course Recommendations</h2>
            {courses_html}
        </div>
    </div>

    <div class="footer">
        OrchestrAI Practice Portal &bull; Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    </div>
</body>
</html>"""
    return html


# ==============================================================================
# FEATURE 8 (cont.) — Upload HTML to GitHub
# ==============================================================================

def save_practice_html_to_github(company: str, role: str, html_content: str) -> str:
    """Upload the rendered HTML practice page to GitHub. Returns public raw URL."""
    file_name = f"{_slugify(company)}_{_slugify(role)}.html"
    file_path = f"frontend/practice/{file_name}"

    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(
            file_path,
            html_content,
            sha,
            f"feat: generate practice portal for {company} {role} — {ts}",
        )
        return _get_github_url(file_path)
    except Exception as exc:
        logger.error("PracticeAgent: save_practice_html_to_github failed — %s", exc)
        return ""


# ==============================================================================
# FEATURE 9 — Save practice sessions YAML
# ==============================================================================

def save_practice_sessions(sessions: list[dict]) -> bool:
    try:
        return write_yaml_to_github(PRACTICE_SESSIONS_FILE, sessions)
    except Exception as exc:
        logger.error("PracticeAgent: save_practice_sessions failed — %s", exc)
        return False


# ==============================================================================
# FEATURE 11 — Logging
# ==============================================================================

def log_agent_activity(action: str, status: str = "success") -> None:
    try:
        append_log_entry({
            "agent": "PracticeAgent",
            "action": action,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        pass


# ==============================================================================
# MAIN — run_practice_agent()
# ==============================================================================

def run_practice_agent() -> list[dict]:
    """
    Execute the full PracticeAgent pipeline.
    Returns list of practice session dicts for ExecutionAgent.
    """
    logger.info("PracticeAgent: Starting...")
    log_agent_activity("PracticeAgent run initiated")

    # ── Load inputs ──
    jobs = read_jobs()
    user = read_user_profile()
    skill_gaps_list = read_skill_gaps()
    resume_text = load_resume_text()

    if not jobs:
        logger.warning("PracticeAgent: No jobs found — skipping.")
        log_agent_activity("Skipped — no jobs in database", "partial")
        return []

    user_name = user.get("name", "Applicant")
    user_skills = user.get("resume_skills", [])
    career_goals = user.get("career_goals", [])

    # Build skill-gap lookup
    gap_lookup = {
        (item.get("company", ""), item.get("role", "")): item
        for item in skill_gaps_list if isinstance(item, dict)
    }

    practice_sessions: list[dict] = []

    for job in jobs:
        if not isinstance(job, dict):
            continue

        company = job.get("company", "Unknown")
        role = job.get("role", "Unknown")
        tech_skills = job.get("technical_skills", [])
        key = (company, role)

        logger.info("PracticeAgent: Generating portal for %s — %s", company, role)

        # Skill gap for this job
        gap_info = gap_lookup.get(key, {})
        missing_skills = gap_info.get("missing_skills", [])

        # -- Feature 1: Interview Q&A --
        qa_pairs = generate_interview_qa(company, role, tech_skills, resume_text, user_skills)

        # -- Feature 2: HR Introduction --
        hr_intro = generate_hr_introduction(user, company, role, resume_text)

        # -- Feature 3: Tamil → English samples --
        translations = _generate_sample_translations()

        # -- Feature 4: Speaking Practice --
        speaking = generate_speaking_practice(role, tech_skills)

        # -- Feature 5: Coding Sheets --
        coding_sheets = generate_coding_sheets(role, tech_skills)

        # -- Feature 6: Project Recommendations --
        projects = generate_project_recommendations(missing_skills, role)

        # -- Feature 7: Course Recommendations --
        courses = generate_course_recommendations(
            missing_skills if missing_skills else tech_skills[:5], role
        )

        # -- Feature 8: Generate HTML --
        html = _render_practice_html(
            company, role, qa_pairs, hr_intro,
            translations, speaking, coding_sheets,
            projects, courses,
        )

        # -- Upload to GitHub --
        practice_link = save_practice_html_to_github(company, role, html)

        # -- Feature 9 & 10: Record session --
        practice_sessions.append({
            "company": company,
            "role": role,
            "practice_link": practice_link,
        })

    # ── Save practice sessions database ──
    if practice_sessions:
        save_practice_sessions(practice_sessions)
        log_agent_activity(f"Generated practice portals for {len(practice_sessions)} internships")

    logger.info("PracticeAgent: Completed successfully.")
    return practice_sessions


# ==============================================================================
# Standalone runner
# ==============================================================================

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_practice_agent()
    print("\n--- PracticeAgent Output ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
