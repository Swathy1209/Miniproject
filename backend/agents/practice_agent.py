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

# Use shared circuit breaker from ai_engine
from backend.utils.ai_engine import safe_llm_call as _cb_llm_call, _is_daily_quota_error

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


def _get_public_url(file_path: str) -> str:
    """Helper to cleanly build the final public URL."""
    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    return f"{base_url}/{file_path}"


import time
def _ai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Send a chat completion request — circuit breaker aware."""
    result = _cb_llm_call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        context=f"practice:{system_prompt[:40]}",
    )
    if result is None:
        return ""
    return result


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
# REAL-TIME INTERACTIVE INTERVIEW COACH  (API-facing functions)
# ==============================================================================

INTERACTIONS_FILE = "database/interview_interactions.yaml"


def _detect_language(text: str) -> str:
    """
    Detect whether the input is Tamil (Unicode range U+0B80–U+0BFF)
    or Romanised Tamil / English.
    Returns 'Tamil' or 'English'.
    """
    tamil_chars = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    # Also catch common romanised Tamil words
    romanised_tamil = [
        "epdi", "enna", "theriyum", "pannuven", "kudukurathu",
        "solluven", "pannrom", "ingae", "avanga", "apparam",
        "nan ", "naan ", "oru ", "enakku", "irukku",
    ]
    text_lower = text.lower()
    has_romanised = any(kw in text_lower for kw in romanised_tamil)

    if tamil_chars > 0 or has_romanised:
        return "Tamil"
    return "English"


def generate_interview_response(company: str, role: str, user_input: str) -> dict:
    """
    Real-time AI interview coaching function.

    1. Detects if input is Tamil (Unicode or Romanised).
    2. If Tamil → translates to professional English first.
    3. Generates:
        - professional_answer: Strong, interview-ready answer.
        - practice_version:    Simplified version for rehearsal.
        - confidence_tips:     2 actionable speaking/confidence tips.
    4. Returns structured dict.

    Raises RuntimeError if Gemini fails.
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty.")

    detected_lang = _detect_language(user_input)

    system = (
        "You are an expert AI interview coach helping candidates prepare for internship interviews. "
        "Your job is to:\n"
        "1. If the user wrote in Tamil (Unicode or Romanised), first convert it to a clear English question.\n"
        "2. Provide a PROFESSIONAL interview answer (3-4 confident, polished sentences) tailored to the company and role.\n"
        "3. Provide a SIMPLIFIED PRACTICE VERSION of the same answer (short, easy to memorise, natural English).\n"
        "4. Provide exactly 2 CONFIDENCE TIPS specific to this type of question.\n\n"
        "Format your response STRICTLY as:\n"
        "PROFESSIONAL_ANSWER: <answer>\n"
        "PRACTICE_VERSION: <simplified version>\n"
        "TIP1: <first tip>\n"
        "TIP2: <second tip>"
    )

    prompt = (
        f"Internship Role: {role}\n"
        f"Company: {company}\n"
        f"Detected Language: {detected_lang}\n\n"
        f"User Input:\n{user_input}\n\n"
        "Generate the professional answer, practice version, and 2 confidence tips."
    )

    raw = _ai_chat(system, prompt, max_tokens=800)
    if not raw:
        raise RuntimeError("Gemini API is unavailable. Please try again in a moment.")

    # ── Parse structured response ─────────────────────────────────────────────
    result = {
        "professional_answer": "",
        "practice_version": "",
        "confidence_tips": [],
        "detected_language": detected_lang,
    }

    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("PROFESSIONAL_ANSWER:"):
            result["professional_answer"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PRACTICE_VERSION:"):
            result["practice_version"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TIP1:"):
            result["confidence_tips"].append(line.split(":", 1)[1].strip())
        elif line.upper().startswith("TIP2:"):
            result["confidence_tips"].append(line.split(":", 1)[1].strip())

    # Fallback: if parsing failed, use raw as professional answer
    if not result["professional_answer"] and raw:
        result["professional_answer"] = raw
        result["practice_version"] = raw
        result["confidence_tips"] = ["Speak clearly and confidently.", "Pause before answering to collect your thoughts."]

    logger.info(
        "PracticeAgent: Real-time response generated for %s — %s [lang=%s]",
        company, role, detected_lang
    )
    return result


def validate_company_role(company: str, role: str) -> bool:
    """
    Security check: verify the company+role exists in the live jobs database
    before generating a response. Prevents abuse of the endpoint.
    Returns True if found, False otherwise.
    """
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", []) if isinstance(data, dict) else []
        company_lower = company.lower()
        role_lower = role.lower()
        for job in jobs:
            if isinstance(job, dict):
                if (job.get("company", "").lower() == company_lower and
                        job.get("role", "").lower() == role_lower):
                    return True
        return False
    except Exception as exc:
        logger.warning("PracticeAgent: validate_company_role failed — %s", exc)
        return True  # Fail open (don't block if DB is unreachable)


def log_interview_interaction(company: str, role: str, user_input: str) -> None:
    """
    Append the user interaction to database/interview_interactions.yaml on GitHub.

    YAML structure:
      - company: NVIDIA
        role:    AI Intern
        user_input: "HR kita epdi sollanum"
        timestamp: 2026-03-02T21:00:00
    """
    try:
        existing = read_yaml_from_github(INTERACTIONS_FILE)
        if isinstance(existing, dict):
            interactions = existing.get("interactions", [])
        elif isinstance(existing, list):
            interactions = existing
        else:
            interactions = []

        interactions.append({
            "company":    company,
            "role":       role,
            "user_input": user_input[:500],   # Truncate for storage
            "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })

        write_yaml_to_github(
            INTERACTIONS_FILE,
            {"interactions": interactions},
        )
        logger.info("PracticeAgent: Interaction logged for %s — %s", company, role)
    except Exception as exc:
        logger.warning("PracticeAgent: log_interview_interaction failed — %s", exc)


# ==============================================================================
# FEATURE 1 — Interview Questions & Answers
# ==============================================================================

def generate_interview_qa(company: str, role: str, skills: list[str],
                          resume_text: str, user_skills: list[str]) -> list[dict]:
    """Generate 10 real, role-specific interview Q&A pairs using Gemini."""
    system = (
        "You are a senior technical interviewer at a top tech company. "
        "Generate exactly 10 highly specific interview questions for the given role and company. "
        "Each answer must be personalized using the candidate's actual skills and resume content. "
        "Make answers detailed, confident, and interview-ready — not generic. "
        "Format strictly as:\nQ1: <question>\nA1: <answer>\nQ2: <question>\nA2: <answer> (and so on)"
    )
    prompt = (
        f"Company: {company}\nRole: {role}\n"
        f"Required Skills: {', '.join(skills)}\n"
        f"Candidate Skills: {', '.join(user_skills)}\n"
        f"Resume Excerpt (use this for personalized answers):\n{resume_text[:2000]}\n\n"
        "Generate 10 interview Q&A pairs. Each answer should be 2-4 sentences, specific, and confident."
    )
    raw = _ai_chat(system, prompt, max_tokens=2000)
    if not raw:
        raise RuntimeError(f"Gemini failed to generate interview Q&A for {company} - {role}")
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
        elif q and line:
            q += " " + line
    if q and a:
        pairs.append({"question": q, "answer": a})
    if not pairs:
        raise RuntimeError(f"Gemini returned unparseable Q&A for {company} - {role}")
    return pairs


# ==============================================================================
# FEATURE 2 — HR Introduction Generator
# ==============================================================================

def generate_hr_introduction(user: dict, company: str, role: str, resume_text: str) -> str:
    """Generate a real personalized HR self-introduction using Gemini."""
    name = user.get("name", "Applicant")
    skills = ", ".join(user.get("resume_skills", []))
    goals = ", ".join(user.get("career_goals", []))
    education = user.get("education", "")
    projects = user.get("projects", [])
    projects_str = ", ".join(projects) if projects else ""

    system = (
        "You are an expert career coach. Write a polished, natural, confident self-introduction "
        "for a job interview HR round. It must sound like a real person speaking — "
        "specific, warm, and professional. Mention real skills and career direction. "
        "Length: exactly 80-100 words. No bullet points, just fluent speech."
    )
    prompt = (
        f"Candidate Name: {name}\n"
        f"Education: {education}\n"
        f"Technical Skills: {skills}\n"
        f"Projects: {projects_str}\n"
        f"Career Goals: {goals}\n"
        f"Applying for: {role} at {company}\n"
        f"Resume Context:\n{resume_text[:1000]}\n\n"
        "Write a compelling 80-100 word interview self-introduction in first person."
    )
    result = _ai_chat(system, prompt, max_tokens=400)
    if not result:
        raise RuntimeError(f"Gemini failed to generate HR introduction for {name} at {company}")
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


def _generate_ai_translations(role: str, company: str, user_skills: list[str]) -> list[dict]:
    """
    Use Gemini to generate 5 real Tamil interview phrases + both English versions,
    tailored to the specific role and company context.
    """
    system = (
        "You are a Tamil-English bilingual interview coach. "
        "Generate 5 real Tamil interview phrases a candidate might naturally say, "
        "then provide two English versions for each: "
        "1) Professional English suitable for a formal job interview "
        "2) Simple practice version for beginners. "
        "Make the Tamil phrases realistic and role-specific. "
        "Format strictly as:\n"
        "TAMIL: <tamil phrase>\n"
        "PROFESSIONAL: <professional english>\n"
        "PRACTICE: <simple english>\n"
        "(repeat for all 5)"
    )
    prompt = (
        f"Role: {role} at {company}\n"
        f"Key Skills: {', '.join(user_skills[:6])}\n\n"
        "Generate 5 Tamil interview phrases with professional and simple English translations."
    )
    raw = _ai_chat(system, prompt, max_tokens=1000)
    if not raw:
        raise RuntimeError("Gemini failed to generate Tamil translations")

    translations = []
    current = {}
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("TAMIL:"):
            if current.get("tamil") and current.get("professional"):
                translations.append(current)
            current = {"tamil": line.split(":", 1)[1].strip()}
        elif line.upper().startswith("PROFESSIONAL:"):
            current["professional"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PRACTICE:"):
            current["practice"] = line.split(":", 1)[1].strip()
    if current.get("tamil") and current.get("professional"):
        translations.append(current)

    if not translations:
        raise RuntimeError("Gemini returned unparseable translations")
    return translations


# ==============================================================================
# FEATURE 4 — English Speaking Practice
# ==============================================================================

def generate_speaking_practice(role: str, company: str, skills: list[str], user_skills: list[str]) -> list[dict]:
    """Generate real, role-specific English speaking practice sentences and tips using Gemini."""
    system = (
        "You are an expert English speaking coach for technical job interviews. "
        "Generate exactly 8 practice sentences tailored to the specific role and company. "
        "Each sentence should sound natural and be something a candidate would actually say. "
        "Then generate 5 actionable confidence tips specific to this role. "
        "Format strictly as:\n"
        "PRACTICE SENTENCES:\n1. ...\n2. ...\n(up to 8)\n\n"
        "CONFIDENCE TIPS:\n1. ...\n2. ...\n(up to 5)"
    )
    prompt = (
        f"Role: {role} at {company}\n"
        f"Required Skills: {', '.join(skills)}\n"
        f"Candidate Skills: {', '.join(user_skills)}\n\n"
        "Generate 8 practice sentences and 5 confidence tips specific to this role."
    )
    raw = _ai_chat(system, prompt, max_tokens=800)
    if not raw:
        raise RuntimeError(f"Gemini failed to generate speaking practice for {role} at {company}")

    sentences, tips = [], []
    mode = None
    for line in raw.split("\n"):
        line = line.strip()
        if "PRACTICE SENTENCES" in line.upper():
            mode = "sentences"
            continue
        if "CONFIDENCE TIPS" in line.upper():
            mode = "tips"
            continue
        cleaned = re.sub(r'^\d+[.)\-]\s*', '', line).strip()
        if cleaned:
            if mode == "sentences":
                sentences.append(cleaned)
            elif mode == "tips":
                tips.append(cleaned)

    if not sentences or not tips:
        raise RuntimeError(f"Gemini returned unparseable speaking practice for {role}")
    return [{"sentences": sentences[:8], "tips": tips[:5]}]


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

def generate_project_recommendations(missing_skills: list[str], role: str, company: str) -> list[dict]:
    """Generate real, specific project ideas for each missing skill using Gemini."""
    if not missing_skills:
        system = "You are a technical mentor. Suggest 2 portfolio projects for a strong candidate."
        prompt = f"Role: {role} at {company}. The candidate has all required skills. Suggest 2 impressive portfolio projects."
        raw = _ai_chat(system, prompt, max_tokens=400)
        if not raw:
            raise RuntimeError("Gemini failed to generate project recommendations")
        return [{"skill": "Portfolio Enhancement", "project": raw.strip()}]

    system = (
        "You are a senior technical mentor. For each missing skill, suggest ONE concrete, "
        "buildable project idea that directly demonstrates that skill. "
        "The project must be specific, realistic, and impressive for a portfolio. "
        "Include tech stack hints. "
        "Format strictly as:\nSkill: <skill name>\nProject: <detailed project idea>\n"
        "(repeat for each skill)"
    )
    prompt = (
        f"Target Role: {role} at {company}\n"
        f"Missing Skills: {', '.join(missing_skills)}\n\n"
        "Suggest one specific buildable project per missing skill."
    )
    raw = _ai_chat(system, prompt, max_tokens=800)
    if not raw:
        raise RuntimeError(f"Gemini failed to generate project recommendations for {role}")

    recommendations = []
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
        raise RuntimeError(f"Gemini returned unparseable project recommendations for {role}")
    return recommendations


# ==============================================================================
# FEATURE 7 — Course Recommendations
# ==============================================================================

def generate_course_recommendations(missing_skills: list[str], role: str, company: str) -> list[dict]:
    """Generate real, specific course and resource links for each missing skill using Gemini."""
    if not missing_skills:
        return []

    system = (
        "You are an expert learning advisor. For each missing skill, recommend 3 specific "
        "free learning resources. Include actual YouTube channel names, real Coursera course titles, "
        "and official documentation links. Be specific — no generic suggestions. "
        "Format strictly as:\nSkill: <skill name>\n- <resource description>\n- <resource description>\n- <resource description>\n"
        "(repeat for each skill)"
    )
    prompt = (
        f"Target Role: {role} at {company}\n"
        f"Skills to learn: {', '.join(missing_skills)}\n\n"
        "For each skill, provide 3 specific free learning resources with real names and links."
    )
    raw = _ai_chat(system, prompt, max_tokens=1200)
    if not raw:
        raise RuntimeError(f"Gemini failed to generate course recommendations for {role}")

    courses = []
    current_skill = ""
    for line in raw.split("\n"):
        line = line.strip()
        if line.lower().startswith("skill:"):
            current_skill = line.split(":", 1)[1].strip()
        elif (line.startswith("-") or line.startswith("•")) and current_skill:
            courses.append({
                "skill": current_skill,
                "recommendation": line.lstrip("-•").strip(),
            })

    if not courses:
        raise RuntimeError(f"Gemini returned unparseable course recommendations for {role}")
    return courses


# ==============================================================================
# FEATURE 8 — Generate Practice HTML Portal
# ==============================================================================

# ==============================================================================
# FEATURE 8 — Generate Practice HTML Portal (PRODUCTION UI)
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
    """Render a beautiful, self-contained HTML practice portal akin to LeetCode/LinkedIn Prep."""
    
    # Validation step: Disallow placeholders/poor rendering.
    required_len = {"qa": 10, "trans": 3, "speak": 1}
    if len(qa_pairs) < required_len["qa"] or not hr_intro or "Placeholder" in hr_intro:
        raise ValueError("Detected incomplete placeholder content. Regenerating...")

    # --- Feature 1: Real Q&A ---
    qa_html = ""
    for i, pair in enumerate(qa_pairs, 1):
        qa_html += f"""
        <div class="card qa-card">
            <div class="qa-header">
                <span class="q-badge">Q{i}</span>
                <span class="q-text">{pair['question']}</span>
            </div>
            <div class="qa-body">
                <div class="a-label">Ideal Response:</div>
                <div class="a-text">{pair['answer']}</div>
            </div>
        </div>"""

    # --- Feature 3: Translations ---
    trans_html = ""
    for idx, t in enumerate(translations, 1):
        trans_html += f"""
        <div class="trans-item">
            <div class="trans-tamil"><span class="lang-tag ta">Tamil</span> "{t['tamil']}"</div>
            <div class="trans-english"><span class="lang-tag en">Pro English</span> {t['professional']}</div>
            <div class="trans-practice"><span class="lang-tag pr">Simple</span> {t['practice']}</div>
        </div>"""

    # --- Feature 4: Speaking ---
    sp_data = speaking[0] if speaking else {"sentences": [], "tips": []}
    sentences_html = "".join(f'<div class="sentence-box">💬 {s}</div>' for s in sp_data.get("sentences", []))
    tips_html = "".join(f'<div class="tip-box">✨ {t}</div>' for t in sp_data.get("tips", []))

    # --- Feature 5 & 6 & 7: Resources ---
    sheets_html = "".join(f'<a href="{cs["url"]}" target="_blank" class="resource-link"><span class="r-icon">💻</span>{cs["name"]}</a>' for cs in coding_sheets)
    projects_html = "".join(f"""
        <div class="proj-card">
            <div class="proj-skill">🎯 Skill Focus: {p["skill"]}</div>
            <div class="proj-desc">{p["project"]}</div>
        </div>""" for p in projects)
    
    # Courses rendering
    courses_html = ""
    current_skill = ""
    for c in courses:
        if c["skill"] != current_skill:
            if current_skill: courses_html += "</div>"
            current_skill = c["skill"]
            courses_html += f'<div class="course-group"><div class="c-skill-title">📘 {current_skill}</div>'
        # Parse YouTube/Coursera links slightly better if possible, mostly just text format
        text = c["recommendation"]
        is_link = "http" in text
        html_val = f'<a href="{text}" target="_blank" class="c-link">🔗 View Resource</a>' if is_link else text
        courses_html += f'<div class="c-item">{html_val}</div>'
    if current_skill: courses_html += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company} | {role} - Interview Prep</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #09090b;
            --bg-card: rgba(255, 255, 255, 0.03);
            --bg-hover: rgba(255, 255, 255, 0.08);
            --accent: #3b82f6;
            --secondary: #8b5cf6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
            --succ: #10b981;
            --warn: #eab308;
            --err: #ef4444;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.1), transparent 30%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.1), transparent 30%);
            background-attachment: fixed;
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}
        
        /* Navbar */
        .navbar {{
            background: rgba(9, 9, 11, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .brand {{ 
            font-family: 'Outfit', sans-serif;
            font-weight: 800; font-size: 1.4rem; 
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #ffffff 30%, #a8a29e 100%); 
            -webkit-background-clip: text; color: transparent; 
        }}
        .nav-role {{ font-weight: 500; color: var(--accent); font-size: 0.95rem; background: rgba(59, 130, 246, 0.1); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(59, 130, 246, 0.2);}}
        
        /* Main Layout */
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            width: 100%;
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 2rem;
            flex: 1;
        }}
        
        /* Sidebar Navigation */
        .sidebar {{
            position: sticky;
            top: 5rem;
            height: max-content;
        }}
        .tab-btn {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.875rem 1rem;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            transition: all 0.2s;
            font-family: inherit;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .tab-btn:hover {{ background-color: rgba(255,255,255,0.03); color: var(--text-main); }}
        .tab-btn.active {{
            background-color: rgba(99, 102, 241, 0.1);
            color: var(--accent);
            font-weight: 600;
        }}
        
        /* Content Area */
        .content-area {{ min-width: 0; }}
        .tab-pane {{ display: none; animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1); }}
        .tab-pane.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        
        h1.page-title {{ 
            font-family: 'Outfit', sans-serif; 
            font-size: 2.5rem; 
            font-weight: 800; 
            margin-bottom: 0.5rem; 
            letter-spacing: -0.5px;
        }}
        p.page-subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 1.05rem; }}

        /* Cards & UI Blocks */
        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{
            border-color: rgba(255,255,255,0.15);
        }}

        /* Q&A Styles */
        .qa-card {{ padding: 0; overflow: hidden; }}
        .qa-header {{ background: rgba(0,0,0,0.2); padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; gap: 1rem; align-items: flex-start; }}
        .q-badge {{ background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: var(--accent); padding: 0.25rem 0.75rem; border-radius: 6px; font-weight: 700; font-size: 0.85rem; }}
        .q-text {{ font-weight: 600; font-size: 1.1rem; line-height: 1.5; color: #fff; }}
        .qa-body {{ padding: 1.5rem; }}
        .a-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 700; margin-bottom: 0.75rem; display: block; }}
        .a-text {{ color: #e2e8f0; line-height: 1.7; }}

        /* Intro Box */
        .intro-box {{ font-size: 1.1rem; line-height: 1.8; color: #e2e8f0; font-style: italic; border-left: 4px solid var(--accent); padding-left: 1.5rem; }}

        /* Translator */
        .trans-item {{ background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; }}
        .trans-item > div {{ margin-bottom: 0.75rem; line-height: 1.5; }}
        .trans-item > div:last-child {{ margin-bottom: 0; }}
        .lang-tag {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; margin-right: 0.75rem; text-transform: uppercase; }}
        .ta {{ background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }}
        .en {{ background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }}
        .pr {{ background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }}

        /* Speaking & Tips */
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        .sentence-box {{ background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border: 1px solid var(--border); }}
        .tip-box {{ background: rgba(245,158,11,0.05); padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem; border: 1px solid rgba(245,158,11,0.2); color: #fef3c7; }}

        /* Resources */
        .resource-link {{ display: flex; align-items: center; gap: 1rem; padding: 1.25rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 1rem; color: var(--text-main); text-decoration: none; font-weight: 500; transition: all 0.2s; }}
        .resource-link:hover {{ border-color: var(--accent); transform: translateX(4px); }}
        .r-icon {{ font-size: 1.5rem; }}
        .proj-card {{ border-left: 3px solid var(--succ); background: var(--bg-card); padding: 1.25rem; border-radius: 0 12px 12px 0; margin-bottom: 1rem; border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
        .proj-skill {{ color: var(--succ); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.5rem; }}
        
        .course-group {{ margin-bottom: 2rem; }}
        .c-skill-title {{ font-size: 1.1rem; font-weight: 600; color: #a5b4fc; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
        .c-item {{ padding: 0.5rem 0; color: #d1d5db; display: flex; gap: 0.5rem; align-items: flex-start; }}
        .c-item::before {{ content: "•"; color: var(--text-muted); }}
        .c-link {{ color: var(--accent); text-decoration: none; font-weight: 500; background: rgba(99,102,241,0.1); padding: 0.25rem 0.75rem; border-radius: 4px; }}
        .c-link:hover {{ background: rgba(99,102,241,0.2); }}

        /* Coach Chat UI */
        .coach-container {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; display: flex; flex-direction: column; height: 600px; }}
        .coach-header {{ background: rgba(255,255,255,0.02); padding: 1.25rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 1rem; }}
        .coach-avatar {{ width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 4px 12px rgba(99,102,241,0.3); }}
        .coach-status {{ display: flex; flex-direction: column; }}
        .coach-name {{ font-weight: 600; font-size: 1rem; }}
        .coach-online {{ font-size: 0.8rem; color: var(--succ); display: flex; align-items: center; gap: 0.3rem; }}
        .coach-online::before {{ content: ''; width: 8px; height: 8px; background: var(--succ); border-radius: 50%; display: inline-block; box-shadow: 0 0 8px var(--succ); }}
        
        .chat-history {{ flex: 1; padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.5rem; }}
        .msg-user {{ align-self: flex-end; background: var(--accent); color: white; padding: 1rem 1.25rem; border-radius: 16px 16px 0 16px; max-width: 80%; line-height: 1.5; }}
        .msg-ai {{ align-self: flex-start; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--text-main); padding: 1.25rem; border-radius: 16px 16px 16px 0; max-width: 90%; width: 100%; }}
        
        .ai-chunk {{ margin-bottom: 1rem; }}
        .ai-chunk:last-child {{ margin-bottom: 0; }}
        .ai-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #a5b4fc; margin-bottom: 0.5rem; }}
        .tips-ul {{ padding-left: 1.25rem; margin-top: 0.5rem; color: #fcd34d; font-size: 0.95rem; }}
        .tips-ul li {{ margin-bottom: 0.25rem; }}
        
        .chat-input-area {{ padding: 1.5rem; border-top: 1px solid var(--border); background: var(--bg-main); display: flex; gap: 1rem; align-items: flex-end; }}
        .chat-textarea {{ flex: 1; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; color: var(--text-main); font-family: inherit; font-size: 0.95rem; resize: none; overflow: hidden; max-height: 150px; outline: none; transition: border-color 0.2s; }}
        .chat-textarea:focus {{ border-color: var(--accent); }}
        .chat-btn {{ background: var(--accent); color: white; border: none; width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; flex-shrink: 0; }}
        .chat-btn:hover:not(:disabled) {{ background: var(--accent-hover); transform: translateY(-2px); }}
        .chat-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .chat-btn svg {{ width: 20px; height: 20px; fill: currentColor; }}

        .typing-indicator {{ display: flex; gap: 4px; padding: 0.5rem 1rem; background: rgba(255,255,255,0.05); border-radius: 16px; width: fit-content; align-self: flex-start; }}
        .dot {{ width: 8px; height: 8px; background: var(--text-muted); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }}
        .dot:nth-child(1) {{ animation-delay: -0.32s; }}
        .dot:nth-child(2) {{ animation-delay: -0.16s; }}
        @keyframes bounce {{ 0%, 80%, 100% {{ transform: scale(0); }} 40% {{ transform: scale(1); }} }}

        @media (max-width: 768px) {{
            .container {{ grid-template-columns: 1fr; padding: 1rem; }}
            .sidebar {{ position: static; display: flex; overflow-x: auto; padding-bottom: 1rem; }}
            .tab-btn {{ white-space: nowrap; width: auto; margin-right: 0.5rem; margin-bottom: 0; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="brand">OrchestrAI Prep</div>
        <div class="nav-role">{company} • {role}</div>
    </nav>

    <div class="container">
        <!-- Sidebar Navigation -->
        <aside class="sidebar">
            <button class="tab-btn active" onclick="switchTab('tab-qa', this)">
                📚 Interview Q&A
            </button>
            <button class="tab-btn" onclick="switchTab('tab-intro', this)">
                🎤 HR Introduction
            </button>
            <button class="tab-btn" onclick="switchTab('tab-trans', this)">
                🌐 Tamil Translaton
            </button>
            <button class="tab-btn" onclick="switchTab('tab-speak', this)">
                🗣️ English Clinic
            </button>
            <button class="tab-btn" onclick="switchTab('tab-res', this)">
                💻 Tools & Courses
            </button>
            <button class="tab-btn" onclick="switchTab('tab-coach', this)" style="border: 1px solid rgba(99,102,241,0.3); background: rgba(99,102,241,0.05); margin-top: 1rem;">
                🤖 Live AI Coach <span style="font-size: 8px; color: #10b981;">●</span>
            </button>
        </aside>

        <!-- Main Content Area -->
        <main class="content-area">
            
            <!-- TAB 1: Q&A -->
            <div id="tab-qa" class="tab-pane active">
                <h1 class="page-title">Role-Specific Questions</h1>
                <p class="page-subtitle">10 personalized questions based on the {role} role and your resume skills.</p>
                {qa_html}
            </div>

            <!-- TAB 2: HR Intro -->
            <div id="tab-intro" class="tab-pane">
                <h1 class="page-title">Personalized Introduction</h1>
                <p class="page-subtitle">A natural 90-second pitch specifically tailored to {company}. Memorize this flow.</p>
                <div class="card intro-box">
                    "{hr_intro}"
                </div>
            </div>

            <!-- TAB 3: Translater -->
            <div id="tab-trans" class="tab-pane">
                <h1 class="page-title">Tamil to English Guide</h1>
                <p class="page-subtitle">Common interview phrases translated into professional and simple English.</p>
                <div class="card" style="padding: 2rem;">
                    {trans_html}
                </div>
            </div>

            <!-- TAB 4: Speaking -->
            <div id="tab-speak" class="tab-pane">
                <h1 class="page-title">Communication Clinic</h1>
                <p class="page-subtitle">Role-specific vocabulary and actionable confidence tips.</p>
                <div class="grid-2">
                    <div>
                        <h3 style="color:#a5b4fc; margin-bottom:1rem; font-size:0.9rem; text-transform:uppercase;">Vocabulary</h3>
                        {sentences_html}
                    </div>
                    <div>
                        <h3 style="color:#fcd34d; margin-bottom:1rem; font-size:0.9rem; text-transform:uppercase;">Delivery Tips</h3>
                        {tips_html}
                    </div>
                </div>
            </div>

            <!-- TAB 5: Resources -->
            <div id="tab-res" class="tab-pane">
                <h1 class="page-title">Preparation Resources</h1>
                <p class="page-subtitle">Hand-picked coding rounds, architectural gaps, and upskilling courses.</p>
                
                <h3 style="margin-bottom: 1rem; margin-top: 0;">Interactive Practice</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 2.5rem;">
                    {sheets_html}
                </div>

                <div class="grid-2">
                    <div>
                        <h3 style="margin-bottom: 1rem;">Missing Skills Projects</h3>
                        {projects_html}
                    </div>
                    <div class="card">
                        <h3 style="margin-bottom: 1.5rem; margin-top: 0;">Recommended Courses</h3>
                        {courses_html}
                    </div>
                </div>
            </div>

            <!-- TAB 6: Live Coach -->
            <div id="tab-coach" class="tab-pane">
                <h1 class="page-title">Interactive Interview Coach</h1>
                <p class="page-subtitle">Ask anything in English or Tamil. Get instant, production-ready responses.</p>
                
                <div class="coach-container">
                    <div class="coach-header">
                        <div class="coach-avatar">🤖</div>
                        <div class="coach-status">
                            <span class="coach-name">OrchestrAI Mentor</span>
                            <span class="coach-online">Online for {company} prep</span>
                        </div>
                    </div>
                    
                    <div class="chat-history" id="chatbox">
                        <div class="msg-ai">
                            Hi! I've analyzed the {role} role at {company} and your resume. What interview question are you struggling with? You can ask me in Tamil too!
                        </div>
                    </div>
                    
                    <div class="chat-input-area">
                        <textarea id="aiInput" class="chat-textarea" rows="1" placeholder="Type a question... (Enter to send)"></textarea>
                        <button id="sendBtn" class="chat-btn" onclick="sendMessage()">
                            <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                        </button>
                    </div>
                </div>
            </div>

        </main>
    </div>

    <script>
        // Tab Switching Logic
        function switchTab(tabId, btnElement) {{
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btnElement.classList.add('active');
            window.scrollTo(0,0);
        }}

        // Auto-resize textarea
        const aiInput = document.getElementById('aiInput');
        aiInput.addEventListener('input', function() {{
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        }});

        aiInput.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                sendMessage();
            }}
        }});

        // Live Chat Logic
        const COMPANY_SLUG = "{_slugify(company)}";
        const ROLE_SLUG = "{_slugify(role)}";
        const API_BASE = window.location.origin;

        async function sendMessage() {{
            const val = aiInput.value.trim();
            if(!val) return;

            const chatbox = document.getElementById('chatbox');
            const btn = document.getElementById('sendBtn');
            
            // Add User Msg
            const uDiv = document.createElement('div');
            uDiv.className = 'msg-user';
            uDiv.textContent = val;
            chatbox.appendChild(uDiv);
            
            aiInput.value = '';
            aiInput.style.height = 'auto';
            btn.disabled = true;
            aiInput.disabled = true;

            // Add Typing Indicator
            const tDiv = document.createElement('div');
            tDiv.className = 'typing-indicator';
            tDiv.id = 'typingBubble';
            tDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
            chatbox.appendChild(tDiv);
            chatbox.scrollTop = chatbox.scrollHeight;

            try {{
                const res = await fetch(`${{API_BASE}}/practice/${{COMPANY_SLUG}}/${{ROLE_SLUG}}/ask`, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{question: val}})
                }});

                document.getElementById('typingBubble').remove();

                if(!res.ok) throw new Error("API Limit Reached or Offline.");
                
                const data = await res.json();
                
                const aiDiv = document.createElement('div');
                aiDiv.className = 'msg-ai';
                
                let tipsHtml = '';
                if(data.confidence_tips && data.confidence_tips.length > 0) {{
                    tipsHtml = '<div class="ai-chunk"><div class="ai-label">⚡ Confidence Tips</div><ul class="tips-ul">' + 
                               data.confidence_tips.map(t => `<li>${{t}}</li>`).join('') + '</ul></div>';
                }}

                aiDiv.innerHTML = `
                    ${{data.detected_language === 'Tamil' ? `<div class="ai-chunk" style="font-size:0.8rem; color:var(--text-muted);"><em>Translated from Tamil</em></div>` : ''}}
                    <div class="ai-chunk">
                        <div class="ai-label">💼 Professional Answer</div>
                        <div style="line-height:1.7;">${{data.professional_answer}}</div>
                    </div>
                    <div class="ai-chunk" style="margin-top:1.5rem; padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.1);">
                        <div class="ai-label">🗣️ Simple Practice Version</div>
                        <div style="line-height:1.7;">${{data.practice_version}}</div>
                    </div>
                    ${{tipsHtml}}
                `;
                chatbox.appendChild(aiDiv);

            }} catch(e) {{
                document.getElementById('typingBubble')?.remove();
                const errDiv = document.createElement('div');
                errDiv.className = 'msg-ai';
                errDiv.style.borderLeft = '3px solid var(--err)';
                errDiv.innerHTML = `<strong>Error:</strong> Failed to reach OrchestrAI backend. Make sure: <br>1. API Server is running.<br>2. You have not hit the Gemini Free Tier 15 RPM limit.`;
                chatbox.appendChild(errDiv);
            }} finally {{
                btn.disabled = false;
                aiInput.disabled = false;
                aiInput.focus();
                chatbox.scrollTop = chatbox.scrollHeight;
            }}
        }}
    </script>
</body>
</html>"""
    return html


# ==============================================================================
# FEATURE 8 (cont.) — Upload HTML to Public Path
# ==============================================================================

def save_practice_html_to_github(company: str, role: str, html_content: str) -> str:
    """Save the rendered HTML practice page locally. Returns public URL."""
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
        return _get_public_url(file_path)
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

        # Respect Gemini RPM limit (15 RPM)
        time.sleep(4)

        logger.info("PracticeAgent: Generating portal for %s — %s", company, role)

        # Skill gap for this job
        gap_info = gap_lookup.get(key, {})
        missing_skills = gap_info.get("missing_skills", [])

        try:
            # -- Feature 1: Interview Q&A --
            qa_pairs = generate_interview_qa(company, role, tech_skills, resume_text, user_skills)
            time.sleep(2)

            # -- Feature 2: HR Introduction --
            hr_intro = generate_hr_introduction(user, company, role, resume_text)
            time.sleep(2)

            # -- Feature 3: Tamil → English (AI-generated, role-specific) --
            translations = _generate_ai_translations(role, company, user_skills)
            time.sleep(2)

            # -- Feature 4: Speaking Practice --
            speaking = generate_speaking_practice(role, company, tech_skills, user_skills)
            time.sleep(2)

            # -- Feature 5: Coding Sheets --
            coding_sheets = generate_coding_sheets(role, tech_skills)

            # -- Feature 6: Project Recommendations --
            projects = generate_project_recommendations(missing_skills, role, company)
            time.sleep(2)

            # -- Feature 7: Course Recommendations --
            courses = generate_course_recommendations(
                missing_skills if missing_skills else tech_skills[:5], role, company
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
            logger.info("PracticeAgent: ✅ Portal done for %s — %s", company, role)

        except Exception as exc:
            logger.error("PracticeAgent: ❌ Failed for %s — %s: %s", company, role, exc)
            log_agent_activity(f"Failed portal for {company} {role}: {exc}", "error")
            continue

    if practice_sessions:
        save_practice_sessions(practice_sessions)
        log_agent_activity(f"Generated practice portals for {len(practice_sessions)} internships")

    logger.info("PracticeAgent: Completed — %d/%d portals generated.", len(practice_sessions), len(jobs))
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
