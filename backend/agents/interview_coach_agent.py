"""
interview_coach_agent.py — AI Interview Coach
OrchestrAI Autonomous Multi-Agent System

PURPOSE:
  For each internship listing, generate a tailored mock interview simulation
  page containing technical, behavioral, coding, and case-study questions.

FLOW:
  1. Read jobs from database/jobs.yaml
  2. Read user profile from database/users.yaml
  3. Use LLM to generate 8+ role-specific questions per job
  4. Build a rich HTML interview practice page
  5. Save HTML to DATA_DIR/frontend/interview/{slug}.html
  6. Serve via Render static mount at /interview/{slug}.html
  7. Save index to database/interview_sessions.yaml
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:40]


def _is_data_role(role: str) -> bool:
    keywords = ["data", "analyst", "analytics", "business analyst", "bi ", "sql"]
    return any(k in role.lower() for k in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Question Generator
# ─────────────────────────────────────────────────────────────────────────────

from backend.utils.ai_engine import safe_llm_call

def _generate_questions(company: str, role: str, skills: list[str], user_skills: list[str]) -> dict:
    """
    Call Gemini to generate 4 categories of interview questions.
    Returns dict with keys: technical, behavioral, coding, case
    """
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
        return fallback

    try:
        def _extract_section(label: str, text: str) -> list[str]:
            pattern = rf"{label}:?\s*\n(.*?)(?=\n[A-Z]+:|\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if not match:
                return []
            lines = match.group(1).strip().split("\n")
            result = []
            for line in lines:
                line = re.sub(r"^\d+\.\s*", "", line).strip()
                if line and len(line) > 5:
                    result.append(line)
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


# ─────────────────────────────────────────────────────────────────────────────
# HTML Page Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_interview_html(
    company: str,
    role: str,
    skills: list[str],
    questions: dict,
    user_name: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%B %d, %Y")
    skill_tags = "".join(
        f'<span style="background:#e8eaf6;color:#3949ab;padding:4px 10px;'
        f'border-radius:20px;font-size:12px;margin:3px;display:inline-block">{s}</span>'
        for s in skills[:8]
    )

    def _q_items(qs: list[str], color: str = "#1a237e") -> str:
        return "".join(
            f'<li style="margin:12px 0;padding:12px 16px;background:white;'
            f'border-left:4px solid {color};border-radius:6px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.06);line-height:1.5">'
            f'<span style="font-weight:600;color:{color}">Q{i+1}.</span> {q}</li>'
            for i, q in enumerate(qs)
        ) if qs else '<li style="color:#999;margin:8px 0">No questions generated.</li>'

    def _code_items(qs: list[str]) -> str:
        if not qs:
            return "<p style='color:#999'>No coding challenges generated.</p>"
        return "".join(
            f'<div style="background:#1e1e2e;border-radius:10px;padding:20px;margin-bottom:14px">'
            f'<p style="color:#cdd6f4;font-size:14px;margin:0;line-height:1.6">'
            f'<span style="color:#89b4fa;font-weight:700">Problem {i+1}:</span> {q}</p>'
            f'<div style="margin-top:12px;background:#181825;border-radius:6px;padding:12px;">'
            f'<span style="color:#6c7086;font-size:12px">// Write your solution here...</span></div>'
            f'</div>'
            for i, q in enumerate(qs)
        )

    # Case section (only for data roles)
    case_section = ""
    if questions.get("case"):
        case_items = "".join(
            f'<div style="background:#fff8e1;border:1px solid #ffd54f;border-radius:8px;'
            f'padding:16px;margin-bottom:12px">'
            f'<span style="font-weight:700;color:#e65100">📊 Case {i+1}:</span> '
            f'<span style="color:#424242">{q}</span></div>'
            for i, q in enumerate(questions["case"])
        )
        case_section = f"""
        <div class="card">
          <h3>📊 Section 4: Case Study Questions</h3>
          <p style="color:#666;font-size:13px;margin-bottom:16px">
            Analytical and data-driven scenario questions for this role
          </p>
          {case_items}
        </div>"""

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
  .ai-feedback{{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px dashed #7c3aed;border-radius:12px;padding:24px;text-align:center}}
  .footer{{text-align:center;color:#555;font-size:12px;padding:30px;border-top:1px solid rgba(255,255,255,0.05)}}
  @media(max-width:600px){{.hero{{padding:40px 20px}}}}
</style>
</head>
<body>

<div class="hero">
  <h1>🎤 Mock Interview Simulation</h1>
  <h2>{role} at <strong>{company}</strong></h2>
  <div>
    <span class="badge">👤 {user_name}</span>
    <span class="badge">📅 {ts}</span>
    <span class="badge">⏱️ ~30 min</span>
  </div>
</div>

<div class="container">
  <!-- Skills -->
  <div class="card">
    <h3>🎯 Skills Being Tested</h3>
    <div style="margin-top:8px">{skill_tags or '<span style="color:#666">General CS skills</span>'}</div>
  </div>

  <!-- Timer -->
  <div class="timer">
    <p style="color:#a78bfa;font-size:13px;margin-bottom:8px;font-weight:600">INTERVIEW TIMER</p>
    <span id="timer">30:00</span>
    <div style="margin-top:12px;display:flex;gap:10px;justify-content:center">
      <button onclick="startTimer()" style="background:#7c3aed;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px">▶ Start</button>
      <button onclick="resetTimer()" style="background:#374151;color:white;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px">↺ Reset</button>
    </div>
  </div>

  <!-- Section 1: Technical -->
  <div class="card">
    <h3>⚙️ Section 1: Technical Interview Questions</h3>
    <p style="color:#666;font-size:13px;margin-bottom:16px">
      Role-specific technical depth questions — answer out loud or in writing
    </p>
    <ul>{_q_items(questions.get('technical', []), '#4f46e5')}</ul>
  </div>

  <!-- Section 2: Coding -->
  <div class="card">
    <h3>💻 Section 2: Coding Challenge</h3>
    <p style="color:#666;font-size:13px;margin-bottom:16px">
      Implement solutions and explain your thought process
    </p>
    {_code_items(questions.get('coding', []))}
  </div>

  <!-- Section 3: Behavioral -->
  <div class="card">
    <h3>🧠 Section 3: Behavioral Questions (STAR Method)</h3>
    <p style="color:#666;font-size:13px;margin-bottom:16px">
      Use the <strong>Situation → Task → Action → Result</strong> framework
    </p>
    <ul>{_q_items(questions.get('behavioral', []), '#059669')}</ul>
  </div>

  {case_section}

  <!-- Log Feedback Form -->
  <div class="card" style="border:2px solid #7c3aed">
    <h3 style="color:#a78bfa">📝 Log Interview Feedback</h3>
    <p style="color:#666;font-size:13px;margin-bottom:18px">
      Submit your feedback after the interview — skill gaps and learning roadmap will
      update automatically in tomorrow's daily report.
    </p>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      <div>
        <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Company</label>
        <input id="fb_company" value="{company}" readonly
          style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;
                 padding:8px 12px;color:#fffffe;font-size:13px"/>
      </div>
      <div>
        <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">Role</label>
        <input id="fb_role" value="{role}" readonly
          style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;
                 padding:8px 12px;color:#fffffe;font-size:13px"/>
      </div>
    </div>

    <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">
      Questions You Faced <span style="color:#666">(one per line)</span>
    </label>
    <textarea id="fb_questions" rows="4" placeholder="Explain gradient descent&#10;What is bias-variance tradeoff&#10;..."
      style="width:100%;background:#0f0e17;border:1px solid #7c3aed;border-radius:6px;
             padding:8px 12px;color:#fffffe;font-size:13px;margin-bottom:14px;
             font-family:Inter,sans-serif;resize:vertical"></textarea>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
      <div>
        <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">
          Confidence Level: <span id="conf_val" style="color:#fffffe">6</span>/10
        </label>
        <input id="fb_confidence" type="range" min="1" max="10" value="6"
          oninput="document.getElementById('conf_val').textContent=this.value"
          style="width:100%;accent-color:#7c3aed"/>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#555">
          <span>Not confident</span><span>Very confident</span>
        </div>
      </div>
      <div>
        <label style="color:#a78bfa;font-size:12px;font-weight:600;display:block;margin-bottom:4px">
          Difficulty Level: <span id="diff_val" style="color:#fffffe">7</span>/10
        </label>
        <input id="fb_difficulty" type="range" min="1" max="10" value="7"
          oninput="document.getElementById('diff_val').textContent=this.value"
          style="width:100%;accent-color:#e65100"/>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#555">
          <span>Easy</span><span>Very hard</span>
        </div>
      </div>
    </div>

    <button onclick="submitFeedback()"
      style="background:#7c3aed;color:white;border:none;padding:12px 28px;
             border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;
             transition:background 0.2s"
      onmouseover="this.style.background='#6d28d9'"
      onmouseout="this.style.background='#7c3aed'">
      📤 Log Interview Feedback
    </button>
    <span id="fb_status" style="margin-left:16px;font-size:13px"></span>
  </div>
</div>

<div class="footer">Generated by OrchestrAI • Interview Coach • {ts}</div>

<script>
// ── Timer ──────────────────────────────────────────────────────────────────
let timerInterval = null;
let seconds = 1800;
function startTimer() {{
  if (timerInterval) return;
  timerInterval = setInterval(() => {{
    if (seconds <= 0) {{ clearInterval(timerInterval); return; }}
    seconds--;
    const m = String(Math.floor(seconds/60)).padStart(2,'0');
    const s = String(seconds%60).padStart(2,'0');
    document.getElementById('timer').textContent = m+':'+s;
  }}, 1000);
}}
function resetTimer() {{
  clearInterval(timerInterval);
  timerInterval = null;
  seconds = 1800;
  document.getElementById('timer').textContent = '30:00';
}}

// ── Feedback Submission ────────────────────────────────────────────────────
async function submitFeedback() {{
  const company    = document.getElementById('fb_company').value.trim();
  const role       = document.getElementById('fb_role').value.trim();
  const rawQ       = document.getElementById('fb_questions').value.trim();
  const confidence = parseInt(document.getElementById('fb_confidence').value);
  const difficulty = parseInt(document.getElementById('fb_difficulty').value);
  const status     = document.getElementById('fb_status');

  if (!rawQ) {{
    status.textContent = '⚠️ Please enter at least one question you faced.';
    status.style.color = 'orange';
    return;
  }}

  const questions_faced = rawQ.split('\\n').map(q => q.trim()).filter(q => q.length > 2);

  const payload = {{ company, role, questions_faced, confidence_level: confidence, difficulty_level: difficulty }};

  status.textContent = '⏳ Saving...';
  status.style.color = '#a78bfa';

  try {{
    const base = window.location.origin;
    const resp = await fetch(base + '/log-feedback', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }});
    const data = await resp.json();
    if (resp.ok && data.status === 'ok') {{
      status.textContent = '✅ ' + data.message;
      status.style.color = '#4ade80';
      document.getElementById('fb_questions').value = '';
    }} else {{
      status.textContent = '❌ ' + (data.message || 'Error saving feedback');
      status.style.color = '#f87171';
    }}
  }} catch(e) {{
    status.textContent = '❌ Network error: ' + e.message;
    status.style.color = '#f87171';
  }}
}}
</script>
</body>
</html>"""



# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def run_interview_coach_agent() -> list[dict]:
    logger.info("InterviewCoachAgent: Starting...")

    jobs_data  = read_yaml_from_github(JOBS_FILE)
    jobs       = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []

    users_data = read_yaml_from_github(USERS_FILE)
    user       = users_data.get("user", {}) if isinstance(users_data, dict) else {}
    user_name  = user.get("name", DEFAULT_USER_NAME)
    user_skills = user.get("resume_skills", DEFAULT_SKILLS)

    from backend.github_yaml_db import DATA_DIR
    interview_dir = os.path.join(DATA_DIR, "frontend", "interview")
    os.makedirs(interview_dir, exist_ok=True)

    base_url = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    index    = []
    generated = 0

    for job in jobs:
        if not isinstance(job, dict):
            continue

        company    = job.get("company", "Unknown")
        role       = job.get("role", "Intern")
        
        # Respect Gemini RPM limit
        time.sleep(3)
        
        job_skills = [str(s) for s in job.get("technical_skills", []) if s]

        try:
            questions = _generate_questions(company, role, job_skills, user_skills)

            html = _build_interview_html(
                company=company,
                role=role,
                skills=job_skills,
                questions=questions,
                user_name=user_name,
            )

            slug = f"{_slugify(company)}_{_slugify(role)}"
            file_name = f"{slug}.html"
            file_path = f"frontend/interview/{file_name}"
            
            _, sha = _get_raw_file(file_path)
            ts = datetime.now(timezone.utc).isoformat()
            _put_raw_file(file_path, html, sha, f"feat(interview): generated for {company} — {ts}")

            interview_url = f"{base_url}/interview/{slug}.html"
            index.append({
                "company":        company,
                "role":           role,
                "interview_link": interview_url,
                "generated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            })
            generated += 1
            logger.info("InterviewCoachAgent: ✓ %s — %s → %s", company, role, interview_url)

        except Exception as exc:
            logger.error("InterviewCoachAgent: Failed for %s %s — %s", company, role, exc)

    # Save index to GitHub
    try:
        write_yaml_to_github(INTERVIEW_INDEX_FILE, {"interview_sessions": index})
    except Exception as exc:
        logger.error("InterviewCoachAgent: Failed to save index — %s", exc)

    try:
        append_log_entry({
            "agent":     "InterviewCoachAgent",
            "action":    f"Generated {generated} interview simulation pages",
            "status":    "success" if generated > 0 else "partial",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        pass

    logger.info("InterviewCoachAgent: Done. %d pages generated.", generated)
    return index


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    results = run_interview_coach_agent()
    print(f"\n✅ Generated {len(results)} interview pages:")
    for r in results:
        print(f"  {r['company']:25s} | {r['role']:35s} | {r['interview_link']}")
