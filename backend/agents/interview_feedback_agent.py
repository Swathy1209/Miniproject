"""
interview_feedback_agent.py — Autonomous Interview Feedback Processor
OrchestrAI Autonomous Multi-Agent System

PURPOSE:
  1. Read interview feedback from database/interview_feedback.yaml
  2. For entries where confidence < 7:
       - Use LLM to map weak topics → formal skill gap names
       - Inject those gaps into database/skill_gap_per_job.yaml
       - Augment learning roadmap with specific study actions
  3. The next daily email automatically reflects updated gaps + roadmap

PIPELINE POSITION:
  Runs BEFORE SkillAgent in the daily pipeline so updated feedback is
  incorporated into that day's skill gap analysis.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.InterviewFeedbackAgent")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client   = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0) if GEMINI_API_KEY else None

FEEDBACK_FILE  = "database/interview_feedback.yaml"
SKILL_GAP_FILE = "database/skill_gap_per_job.yaml"

CONFIDENCE_THRESHOLD = 7   # below this → skill gaps need updating


# ─────────────────────────────────────────────────────────────────────────────
# LLM — map weak topics → formal skill names + roadmap steps
# ─────────────────────────────────────────────────────────────────────────────

def _map_topics_to_skills(topics: list[str], role: str) -> tuple[list[str], list[str]]:
    """
    Use LLM to convert raw interview topics into:
     - formal skill gap names (skills_needed)
     - concrete roadmap steps to close them

    Returns (skills_needed, roadmap_steps)
    """
    fallback_skills = [f"{t.title()} Knowledge" for t in topics[:3]]
    fallback_roadmap = [f"Study {t}" for t in topics[:3]]

    if not openai_client or not topics:
        return fallback_skills, fallback_roadmap

    topics_str = "\n".join(f"- {t}" for t in topics)
    prompt = (
        f"A candidate struggled with these topics in a {role} interview:\n{topics_str}\n\n"
        "Reply in EXACTLY this format, no extra text:\n"
        "SKILLS:\n1. [formal skill name]\n2. [formal skill name]\n\n"
        "ROADMAP:\n1. [specific study action]\n2. [specific study action]\n3. [specific study action]"
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()

        def _parse_section(label: str) -> list[str]:
            import re
            m = re.search(rf"{label}:?\s*\n(.*?)(?=\n[A-Z]+:|\Z)", raw, re.DOTALL | re.IGNORECASE)
            if not m:
                return []
            items = []
            for line in m.group(1).strip().split("\n"):
                line = re.sub(r"^\d+\.\s*", "", line).strip()
                if line and len(line) > 3:
                    items.append(line)
            return items[:4]

        skills  = _parse_section("SKILLS")  or fallback_skills
        roadmap = _parse_section("ROADMAP") or fallback_roadmap
        return skills, roadmap

    except Exception as exc:
        logger.warning("InterviewFeedbackAgent: LLM mapping failed — %s", exc)
        return fallback_skills, fallback_roadmap


# ─────────────────────────────────────────────────────────────────────────────
# Core: update skill_gap_per_job.yaml with feedback-derived gaps
# ─────────────────────────────────────────────────────────────────────────────

def _merge_gaps(existing: list[str], new_skills: list[str]) -> list[str]:
    existing_lower = {s.lower() for s in existing}
    merged = list(existing)
    for s in new_skills:
        if s.lower() not in existing_lower:
            merged.append(s)
            existing_lower.add(s.lower())
    return merged


def _merge_roadmap(existing: list[str], new_steps: list[str]) -> list[str]:
    existing_lower = {s.lower() for s in existing}
    merged = list(existing)
    for s in new_steps:
        if s.lower() not in existing_lower:
            merged.append(s)
            existing_lower.add(s.lower())
    return merged[:10]  # cap at 10 roadmap steps


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def run_interview_feedback_agent() -> dict:
    """
    Process all interview feedback entries.
    For entries with confidence < CONFIDENCE_THRESHOLD:
      - Map weak topics → skill names + roadmap steps via LLM
      - Inject into skill_gap_per_job.yaml for that (company, role)
    """
    logger.info("InterviewFeedbackAgent: Starting...")

    # ── Read feedback ─────────────────────────────────────────────────────────
    feedback_raw = read_yaml_from_github(FEEDBACK_FILE)
    feedbacks: list[dict] = []
    if isinstance(feedback_raw, dict):
        feedbacks = feedback_raw.get("interview_feedback", [])
    elif isinstance(feedback_raw, list):
        feedbacks = feedback_raw

    if not feedbacks:
        logger.info("InterviewFeedbackAgent: No feedback entries found — skipping.")
        return {"status": "skip", "reason": "no feedback"}

    # Only process weak entries
    weak = [f for f in feedbacks if isinstance(f, dict) and int(f.get("confidence", 10)) < CONFIDENCE_THRESHOLD]
    logger.info("InterviewFeedbackAgent: %d total feedback entries, %d below confidence threshold (%d).",
                len(feedbacks), len(weak), CONFIDENCE_THRESHOLD)

    if not weak:
        logger.info("InterviewFeedbackAgent: All feedback shows high confidence — no updates needed.")
        return {"status": "ok", "updates": 0}

    # ── Read current skill gaps ───────────────────────────────────────────────
    gap_data = read_yaml_from_github(SKILL_GAP_FILE) or {}
    analyses: list[dict] = gap_data.get("job_skill_analysis", [])
    # Build mutable lookup keyed by (company, role)
    gap_lookup: dict[tuple, dict] = {
        (a.get("company", ""), a.get("role", "")): a
        for a in analyses if isinstance(a, dict)
    }

    updates = 0
    update_log = []

    for entry in weak:
        company    = entry.get("company", "")
        role       = entry.get("role", "")
        topics     = entry.get("questions_faced", entry.get("topics", []))
        confidence = int(entry.get("confidence", entry.get("confidence_level", 5)))
        difficulty = int(entry.get("difficulty", entry.get("difficulty_level", 5)))

        if not topics:
            continue

        logger.info("InterviewFeedbackAgent: Processing %s / %s — confidence=%d topics=%s",
                    company, role, confidence, topics)

        # Map topics → formal skill gaps + roadmap via LLM
        new_skills, new_roadmap = _map_topics_to_skills(topics, role)

        key = (company, role)
        if key in gap_lookup:
            entry_gap = gap_lookup[key]
            entry_gap["missing_skills"] = _merge_gaps(
                entry_gap.get("missing_skills", []), new_skills
            )
            entry_gap["roadmap"] = _merge_roadmap(
                entry_gap.get("roadmap", []), new_roadmap
            )
            entry_gap["feedback_updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            entry_gap["feedback_note"] = (
                f"Confidence {confidence}/10 on: {', '.join(str(t) for t in topics[:3])}"
            )
        else:
            # No existing gap entry for this role — create one from feedback
            gap_lookup[key] = {
                "company":       company,
                "role":          role,
                "missing_skills": new_skills,
                "roadmap":        new_roadmap,
                "feedback_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "feedback_note": f"Confidence {confidence}/10 on: {', '.join(str(t) for t in topics[:3])}",
            }

        update_log.append(f"{company} / {role}: added {len(new_skills)} skills, {len(new_roadmap)} roadmap steps")
        updates += 1

    # ── Write updated skill gaps back ─────────────────────────────────────────
    if updates > 0:
        updated_analyses = list(gap_lookup.values())
        try:
            write_yaml_to_github(SKILL_GAP_FILE, {"job_skill_analysis": updated_analyses})
            logger.info("InterviewFeedbackAgent: Updated skill_gap_per_job.yaml for %d entries.", updates)
        except Exception as exc:
            logger.error("InterviewFeedbackAgent: Failed to write skill gaps — %s", exc)
            return {"status": "error", "updates": 0}

    # ── Log ───────────────────────────────────────────────────────────────────
    try:
        append_log_entry({
            "agent":     "InterviewFeedbackAgent",
            "action":    f"Processed {len(feedbacks)} feedback entries, updated {updates} skill gap records",
            "details":   "; ".join(update_log),
            "status":    "completed",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        pass

    logger.info("InterviewFeedbackAgent: Done. %d skill gap records updated.", updates)
    return {"status": "ok", "updates": updates, "details": update_log}


# ─────────────────────────────────────────────────────────────────────────────
# API helper — called by the /log-feedback FastAPI endpoint
# ─────────────────────────────────────────────────────────────────────────────

def append_feedback_entry(entry: dict) -> bool:
    """
    Append one feedback record to database/interview_feedback.yaml on GitHub.
    Called from main.py POST /log-feedback endpoint.
    """
    try:
        existing = read_yaml_from_github(FEEDBACK_FILE)
        feedbacks: list = []
        if isinstance(existing, dict):
            feedbacks = existing.get("interview_feedback", [])
        elif isinstance(existing, list):
            feedbacks = existing

        entry["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        feedbacks.append(entry)
        write_yaml_to_github(FEEDBACK_FILE, {"interview_feedback": feedbacks})
        logger.info("InterviewFeedbackAgent: Feedback logged for %s / %s",
                    entry.get("company"), entry.get("role"))
        return True
    except Exception as exc:
        logger.error("InterviewFeedbackAgent: append_feedback_entry failed — %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_interview_feedback_agent()
    print(f"\n{'='*55}")
    print(f"Status  : {result.get('status')}")
    print(f"Updates : {result.get('updates', 0)}")
    for d in result.get("details", []):
        print(f"  → {d}")
