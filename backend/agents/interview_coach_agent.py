"""
interview_coach_agent.py — AI Interview Coach
OrchestrAI Autonomous Multi-Agent System
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.InterviewCoachAgent")

JOBS_FILE             = "database/jobs.yaml"
USERS_FILE            = "database/users.yaml"
INTERVIEW_INDEX_FILE  = "database/interview_sessions.yaml"

DEFAULT_USER_NAME   = "Swathy G"
DEFAULT_SKILLS      = ["Python", "Machine Learning", "SQL", "Data Analysis"]

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]

def _is_data_role(role: str) -> bool:
    keywords = ["data", "analyst", "analytics", "business analyst", "bi ", "sql"]
    return any(k in role.lower() for k in keywords)

from backend.utils.ai_engine import safe_llm_call

def _generate_questions(company: str, role: str, skills: list[str], user_skills: list[str]) -> dict:
    skills_str = ", ".join(skills[:8]) if skills else "Python, ML, SQL"
    user_skills_str = ", ".join(user_skills[:6]) if user_skills else "Python"
    include_case = _is_data_role(role)

    prompt = f"""You are a senior interviewer at {company}.
Generate realistic interview questions for the role: {role}
Required skills: {skills_str}
Candidate has: {user_skills_str}

Return EXACTLY this format (no extra text):
TECHNICAL:
1. [question]
2. [question]
3. [question]

BEHAVIORAL:
1. [question]
2. [question]
3. [question]

CODING:
1. [coding problem title] — [brief description]
2. [coding problem title] — [brief description]

{"CASE:" if include_case else ""}
{"1. [data/business case scenario]" if include_case else ""}
{"2. [data/business case scenario]" if include_case else ""}
"""

    fallback = {
        "technical": [
            f"Explain your experience with {skills[0] if skills else 'Python'} and how you've used it in projects.",
            f"How would you approach building a {role.replace(' Intern', '')} pipeline from scratch?",
            f"Describe a challenging technical problem you solved using {skills[1] if len(skills) > 1 else 'Machine Learning'}.",
        ],
        "behavioral": [
            "Tell me about a time you had to learn a new technology quickly.",
            "Describe a situation where you had to work under tight deadlines.",
            "Give an example of a project where you collaborated with a team.",
        ],
        "coding": [
            f"Implement a function to {skills[0].lower() if skills else 'clean'} a dataset and handle missing values",
            "Write an efficient algorithm for binary search and analyze its time complexity",
        ],
        "case": [
            f"Given a dataset of {company}'s user behavior, how would you identify churn patterns?",
            "A/B test results show 10% lift in metric X but 5% drop in metric Y — what's your recommendation?",
        ] if include_case else [],
    }

    raw = safe_llm_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
        context=f"interview:{company[:20]}"
    )

    if not raw:
        logger.warning(f"InterviewCoachAgent: No LLM response for {company}. Using template fallback.")
        return fallback

    try:
        def _extract_section(label: str, text: str) -> list[str]:
            pattern = rf"{label}:?\s*\n(.*?)(?=\n[A-Z]+:|\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if not match: return []
            lines = match.group(1).strip().split("\n")
            result = []
            for line in lines:
                line = re.sub(r"^\d+\.\s*", "", line).strip()
                if line and len(line) > 5: result.append(line)
            return result[:3]

        return {
            "technical": _extract_section("TECHNICAL", raw) or fallback["technical"],
            "behavioral": _extract_section("BEHAVIORAL", raw) or fallback["behavioral"],
            "coding":    _extract_section("CODING", raw) or fallback["coding"],
            "case":      _extract_section("CASE", raw) if include_case else [],
        }
    except Exception as exc:
        logger.warning("InterviewCoachAgent: Parsing failed for %s — %s. Using fallback.", role, exc)
        return fallback

def _build_interview_html(company: str, role: str, skills: list[str], questions: dict, user_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%B %d, %Y")
    skill_tags = "".join(f'<span style="background:#e8eaf6;color:#3949ab;padding:4px 10px;border-radius:20px;font-size:12px;margin:3px;display:inline-block">{s}</span>' for s in skills[:8])

    def _q_items(qs: list[str], color: str = "#1a237e") -> str:
        return "".join(f'<li style="margin:12px 0;padding:12px 16px;background:white;border-left:4px solid {color};border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.06);line-height:1.5"><span style="font-weight:600;color:{color}">Q{i+1}.</span> {q}</li>' for i, q in enumerate(qs)) if qs else '<li style="color:#999;margin:8px 0">No questions generated.</li>'

    def _code_items(qs: list[str]) -> str:
        if not qs: return "<p style='color:#999'>No coding challenges generated.</p>"
        return "".join(f'<div style="background:#1e1e2e;border-radius:10px;padding:20px;margin-bottom:14px"><p style="color:#cdd6f4;font-size:14px;margin:0;line-height:1.6"><span style="color:#89b4fa;font-weight:700">Problem {i+1}:</span> {q}</p><div style="margin-top:12px;background:#181825;border-radius:6px;padding:12px;"><span style="color:#6c7086;font-size:12px">// Write your solution here...</span></div></div>' for i, q in enumerate(qs))

    case_section = ""
    if questions.get("case"):
        case_items = "".join(f'<div style="background:#fff8e1;border:1px solid #ffd54f;border-radius:8px;padding:16px;margin-bottom:12px"><span style="font-weight:700;color:#e65100">📊 Case {i+1}:</span> <span style="color:#424242">{q}</span></div>' for i, q in enumerate(questions["case"]))
        case_section = f'<div class="card"><h3>📊 Section 4: Case Study Questions</h3><p style="color:#666;font-size:13px;margin-bottom:16px">Analytical and data-driven scenario questions for this role</p>{case_items}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Mock Interview — {role} at {company}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Inter',sans-serif;background:#0f0e17;color:#fffffe;min-height:100vh}}
  .hero{{background:linear-gradient(135deg,#7c3aed,#4f46e5,#2563eb);padding:50px 40px;text-align:center}}
  .hero h1{{font-size:28px;font-weight:700;margin-bottom:8px}}
  .hero h2{{font-size:16px;font-weight:400;opacity:0.85;margin-bottom:16px}}
  .badge{{background:rgba(255,255,255,0.2);padding:6px 16px;border-radius:20px;font-size:12px;display:inline-block;margin:4px}}
  .container{{max-width:860px;margin:0 auto;padding:40px 20px}}
  .card{{background:#1a1a2e;border-radius:14px;padding:28px;margin-bottom:24px;border:1px solid rgba(255,255,255,0.08)}}
  .card h3{{color:#a78bfa;font-size:17px;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.1)}}
  ul{{list-style:none;padding:0}}
  .timer{{background:#16213e;border:2px solid #7c3aed;border-radius:12px;padding:16px;text-align:center;margin-bottom:24px}}
  .timer span{{font-size:36px;font-weight:700;color:#a78bfa;font-variant-numeric:tabular-nums}}
  .footer{{text-align:center;color:#555;font-size:12px;padding:30px;border-top:1px solid rgba(255,255,255,0.05)}}
</style>
</head>
<body>
<div class="hero">
  <h1>🎤 Mock Interview Simulation</h1>
  <h2>{role} at <strong>{company}</strong></h2>
  <div><span class="badge">👤 {user_name}</span><span class="badge">📅 {ts}</span><span class="badge">⏱️ ~30 min</span></div>
</div>
<div class="container">
  <div class="card"><h3>🎯 Skills Being Tested</h3><div style="margin-top:8px">{skill_tags or '<span style="color:#666">General CS skills</span>'}</div></div>
  <div class="timer"><p style="color:#a78bfa;font-size:13px;margin-bottom:8px;font-weight:600">INTERVIEW TIMER</p><span id="timer">30:00</span><div style="margin-top:12px;display:flex;gap:10px;justify-content:center"><button onclick="startTimer()" style="background:#7c3aed;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px">▶ Start</button><button onclick="resetTimer()" style="background:#374151;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px">↺ Reset</button></div></div>
  <div class="card"><h3>⚙️ Section 1: Technical Interview Questions</h3><ul>{_q_items(questions.get('technical', []), '#4f46e5')}</ul></div>
  <div class="card"><h3>💻 Section 2: Coding Challenge</h3>{_code_items(questions.get('coding', []))}</div>
  <div class="card"><h3>🧠 Section 3: Behavioral Questions</h3><ul>{_q_items(questions.get('behavioral', []), '#059669')}</ul></div>
  {case_section}
  <div class="card" style="border:2px solid #7c3aed">
    <h3 style="color:#a78bfa">📝 Log Interview Feedback</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      <div><label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Company</label><input id="fb_company" value="{company}" readonly style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;padding:8px 12px;color:#fffffe;font-size:13px"/></div>
      <div><label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Role</label><input id="fb_role" value="{role}" readonly style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;padding:8px 12px;color:#fffffe;font-size:13px"/></div>
    </div>
    <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Questions You Faced</label>
    <textarea id="fb_questions" rows="4" style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;padding:8px 12px;color:#fffffe;font-size:13px;margin-bottom:14px;font-family:Inter,sans-serif;resize:vertical"></textarea>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
      <div><label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Confidence: <span id="conf_val">6</span>/10</label><input id="fb_confidence" type="range" min="1" max="10" value="6" oninput="document.getElementById('conf_val').textContent=this.value" style="width:100%;accent-color:#7c3aed"/></div>
      <div><label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Difficulty: <span id="diff_val">7</span>/10</label><input id="fb_difficulty" type="range" min="1" max="10" value="7" oninput="document.getElementById('diff_val').textContent=this.value" style="width:100%;accent-color:#e65100"/></div>
    </div>
    <button onclick="submitFeedback()" style="background:#7c3aed;color:white;border:none;padding:12px 28px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;">📤 Log Interview Feedback</button>
    <span id="fb_status" style="margin-left:16px;font-size:13px"></span>
  </div>
</div>
<div class="footer">Generated by OrchestrAI • Interview Coach • {ts}</div>
<script>
let timerInterval = null; let seconds = 1800;
function startTimer() {{ if (timerInterval) return; timerInterval = setInterval(() => {{ if (seconds <= 0) {{ clearInterval(timerInterval); return; }} seconds--; const m = String(Math.floor(seconds/60)).padStart(2,'0'); const s = String(seconds%60).padStart(2,'0'); document.getElementById('timer').textContent = m+':'+s; }}, 1000); }}
function resetTimer() {{ clearInterval(timerInterval); timerInterval = null; seconds = 1800; document.getElementById('timer').textContent = '30:00'; }}
async function submitFeedback() {{
  const company = document.getElementById('fb_company').value; const role = document.getElementById('fb_role').value; const rawQ = document.getElementById('fb_questions').value; 
  const status = document.getElementById('fb_status'); if (!rawQ) {{ status.textContent = '⚠️ Enter questions.'; return; }}
  const payload = {{ company, role, questions_faced: rawQ.split('\\n'), confidence_level: parseInt(document.getElementById('fb_confidence').value), difficulty_level: parseInt(document.getElementById('fb_difficulty').value) }};
  status.textContent = '⏳ Saving...';
  try {{
    const resp = await fetch(window.location.origin + '/log-feedback', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
    if (resp.ok) {{ status.textContent = '✅ Saved!'; }} else {{ status.textContent = '❌ Error'; }}
  }} catch(e) {{ status.textContent = '❌ Network error'; }}
}}
</script>
</body></html>"""

def run_interview_coach_agent() -> list[dict]:
    from backend.utils.ai_engine import is_all_quota_exhausted
    logger.info("InterviewCoachAgent: Starting...")
    jobs_data = read_yaml_from_github(JOBS_FILE)
    jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []
    user_data = read_yaml_from_github(USERS_FILE)
    user = user_data.get("user", {}) if isinstance(user_data, dict) else {}
    user_name = user.get("name", DEFAULT_USER_NAME)
    user_skills = user.get("resume_skills", DEFAULT_SKILLS)

    existing_data = read_yaml_from_github(INTERVIEW_INDEX_FILE)
    index = existing_data.get("interview_sessions", []) if isinstance(existing_data, dict) else []
    existing_keys = {(s.get("company"), s.get("role")) for s in index}

    jobs_to_process = jobs[-30:]
    new_entries = 0
    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://miniproject-bye9.onrender.com")

    for job in jobs_to_process:
        company = job.get("company", "Unknown")
        role = job.get("role", "Intern")
        if (company, role) in existing_keys:
            logger.info("InterviewCoachAgent: Skipping %s - %s", company, role)
            continue
        if is_all_quota_exhausted():
            logger.warning("InterviewCoachAgent: Quota exhausted.")
            break
        time.sleep(3)
        try:
            questions = _generate_questions(company, role, job.get("technical_skills", []), user_skills)
            html = _build_interview_html(company, role, job.get("technical_skills", []), questions, user_name)
            slug = f"{_slugify(company)}_{_slugify(role)}"
            file_path = f"frontend/interview/{slug}.html"
            _, sha = _get_raw_file(file_path)
            _put_raw_file(file_path, html, sha, f"feat: interview for {company}")
            index.append({"company": company, "role": role, "interview_link": f"{base_url}/interview/{slug}.html", "generated_at": datetime.now(timezone.utc).isoformat()})
            new_entries += 1
            logger.info("InterviewCoachAgent: ✓ %s", company)
        except Exception as exc:
            logger.error("InterviewCoachAgent: Failed for %s: %s", company, exc)

    if new_entries > 0:
        write_yaml_to_github(INTERVIEW_INDEX_FILE, {"interview_sessions": index})
        append_log_entry({"agent": "InterviewCoachAgent", "action": f"Generated {new_entries} pages", "status": "success", "timestamp": datetime.now(timezone.utc).isoformat()})
    return index

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run_interview_coach_agent()
