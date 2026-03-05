"""
career_strategy_agent.py — AI Career Coach Agent
OrchestrAI Autonomous Multi-Agent System

PURPOSE:
  Generate an AI-driven weekly career strategy by analyzing:
    - Skill gaps (frequency-ranked from all job analyses)
    - Internship match scores (top 5 with score > 70)
    - Portfolio quality (project count, demo links)
    - Interview practice progress (sessions completed)
    - User career goals

  Acts as a personal AI career coach that delivers a
  concrete, prioritized weekly action plan.

PIPELINE:
  Step 1 — Load user goals        (database/users.yaml)
  Step 2 — Analyze skill gaps     (database/skill_gap_per_job.yaml)
  Step 3 — Analyze opportunities  (database/opportunity_scores.yaml)
  Step 4 — Analyze portfolio      (database/portfolio.yaml)
  Step 5 — Analyze practice       (database/practice_sessions.yaml)
  Step 6 — Generate strategy      (Gemini 2.0 Flash LLM)
  Step 7 — Save strategy          (database/career_strategy.yaml)
  Step 8 — Log activity           (database/agent_logs.yaml)
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.CareerStrategyAgent")

# ── LLM Client ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0) if GEMINI_API_KEY else None

# ── Data File Paths ───────────────────────────────────────────────────────────
USERS_FILE      = "database/users.yaml"
SKILL_GAPS_FILE = "database/skill_gap_per_job.yaml"
SCORES_FILE     = "database/opportunity_scores.yaml"
PORTFOLIO_FILE  = "database/portfolio.yaml"
PRACTICE_FILE   = "database/practice_sessions.yaml"
STRATEGY_FILE   = "database/career_strategy.yaml"

# ── Default user data for when YAML has no entries yet ────────────────────────
DEFAULT_GOALS = ["Data Engineering Internship", "ML Engineering Internship"]

# ==============================================================================
# STEP 1 — Identify Target Career Goal
# ==============================================================================
def _load_career_goals() -> tuple[str, list[str]]:
    """Extract career_goals from users.yaml. Returns (primary_goal, all_goals)."""
    try:
        data = read_yaml_from_github(USERS_FILE) or {}
        user = data.get("user", {})
        goals = user.get("career_goals", DEFAULT_GOALS)
        if isinstance(goals, list) and goals:
            return str(goals[0]), [str(g) for g in goals]
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Could not load user goals - %s", exc)
    return DEFAULT_GOALS[0], DEFAULT_GOALS


# ==============================================================================
# STEP 2 — Analyze Skill Gaps (frequency-ranked across all jobs)
# ==============================================================================
def _analyze_skill_gaps() -> tuple[list[str], int, dict[str, int]]:
    """
    Returns:
        top_missing: list of top missing skills by frequency
        total_gaps:  total number of unique missing skills
        freq_map:    {skill: frequency_count}
    """
    try:
        data = read_yaml_from_github(SKILL_GAPS_FILE) or {}
        job_analyses = data.get("job_skill_analysis", [])
        if not isinstance(job_analyses, list):
            return [], 0, {}
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Could not load skill gaps - %s", exc)
        return [], 0, {}

    freq: Counter = Counter()
    for job in job_analyses:
        for skill in job.get("missing_skills", []):
            if skill:
                freq[str(skill).strip()] += 1

    sorted_skills = freq.most_common(10)
    top_missing = [s for s, _ in sorted_skills[:5]]
    freq_map = dict(sorted_skills)

    logger.info("CareerStrategyAgent: Found %d unique missing skills. Top: %s", len(freq), top_missing)
    return top_missing, len(freq), freq_map


# ==============================================================================
# STEP 3 — Analyze Opportunity Scores (top 5 with match_score > 70)
# ==============================================================================
def _analyze_opportunities() -> tuple[list[str], list[dict]]:
    """Returns (top_job_strings, top_job_dicts) for match_score > 70."""
    try:
        data = read_yaml_from_github(SCORES_FILE)
        scores_list = data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Could not load opportunity scores - %s", exc)
        return [], []

    qualified = [
        s for s in scores_list
        if isinstance(s, dict) and s.get("match_score", 0) > 70
    ]
    qualified.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    top = qualified[:5]

    top_strings = [
        f"{s.get('role', 'Role')} at {s.get('company', 'Company')} "
        f"(Score: {s.get('match_score', 0)}/100, {s.get('selection_probability', 'Unknown')} probability)"
        for s in top
    ]
    logger.info("CareerStrategyAgent: %d top opportunities (score > 70) found.", len(top))
    return top_strings, top


# ==============================================================================
# STEP 4 — Analyze Portfolio Strength
# ==============================================================================
def _analyze_portfolio() -> tuple[int, str, list[str]]:
    """
    Returns:
        num_projects:   count of projects in portfolio
        strength_label: human-readable strength description
        recommendations: list of improvement suggestions
    """
    try:
        data = read_yaml_from_github(PORTFOLIO_FILE) or {}
        portfolio = data.get("portfolio", {})
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Could not load portfolio - %s", exc)
        portfolio = {}

    projects = portfolio.get("projects", [])
    num_projects = len(projects) if isinstance(projects, list) else 0

    # Count demo links
    demo_count = sum(
        1 for p in projects
        if isinstance(p, dict) and p.get("demo_url") and p.get("demo_url") != "#"
    )

    # Evaluate strength
    recs = []
    if num_projects == 0:
        strength = "No portfolio yet"
        recs.append("Start by building 1 end-to-end data pipeline project to showcase on GitHub")
        recs.append("Add a README with project description, tech stack, and setup instructions")
    elif num_projects < 4:
        strength = f"Early-stage ({num_projects} project{'s' if num_projects > 1 else ''})"
        recs.append(f"Build {4 - num_projects} more project(s) to reach a competitive portfolio of 4+")
        if demo_count == 0:
            recs.append("Deploy at least 1 project with a live demo link (Streamlit, Hugging Face Spaces, etc.)")
    elif num_projects < 7:
        strength = f"Solid ({num_projects} projects)"
        if demo_count < 2:
            recs.append("Add live demo links to your top 2 projects to impress recruiters")
    else:
        strength = f"Strong ({num_projects} projects, {demo_count} with live demos)"

    logger.info("CareerStrategyAgent: Portfolio — %s", strength)
    return num_projects, strength, recs


# ==============================================================================
# STEP 5 — Analyze Interview Practice Progress
# ==============================================================================
def _analyze_practice() -> tuple[int, str, Optional[str]]:
    """
    Returns:
        sessions_count: number of practice sessions generated
        progress_label: human-readable progress description
        recommendation: specific advice string or None
    """
    try:
        data = read_yaml_from_github(PRACTICE_FILE)
        sessions = data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Could not load practice sessions - %s", exc)
        sessions = []

    count = len(sessions)
    if count == 0:
        label = "Not started"
        rec = "Start your first mock interview session — even 30 minutes of practice makes a measurable difference"
    elif count < 3:
        label = f"Beginner ({count} session{'s' if count > 1 else ''})"
        rec = f"Complete {5 - count} more practice sessions to reach a confident interview-ready level"
    elif count < 5:
        label = f"Developing ({count} sessions)"
        rec = "Push through 2 more sessions; focus on behavioral questions (STAR method)"
    else:
        label = f"Interview-ready ({count} sessions)"
        rec = None

    logger.info("CareerStrategyAgent: Practice — %s", label)
    return count, label, rec


# ==============================================================================
# STEP 6 — Generate Career Strategy via LLM
# ==============================================================================
def _generate_strategy(
    goal: str,
    all_goals: list[str],
    top_skills: list[str],
    skill_freq: dict[str, int],
    top_jobs: list[str],
    portfolio_num: int,
    portfolio_strength: str,
    portfolio_recs: list[str],
    practice_count: int,
    practice_label: str,
    practice_rec: Optional[str],
) -> list[str]:
    """Call Gemini 2.0 Flash to generate a personalized 5-7 step weekly plan."""

    # Fallback if LLM unavailable
    if not openai_client:
        actions = []
        if top_skills:
            actions.append(f"Study the top missing skills this week: {', '.join(top_skills[:3])}")
        if top_jobs:
            actions.append(f"Apply immediately to your best match: {top_jobs[0]}")
        if portfolio_recs:
            actions.append(portfolio_recs[0])
        if practice_rec:
            actions.append(practice_rec)
        actions.append("Update your LinkedIn profile with your latest project")
        return actions[:7]

    # Format skill context
    skill_context = ", ".join(
        f"{s} (needed in {freq} roles)" for s, freq in list(skill_freq.items())[:5]
    )

    prompt = f"""You are an expert AI career coach. Generate a highly personalized weekly action plan.

=== CANDIDATE PROFILE ===
Primary Career Goal: {goal}
All Career Goals: {', '.join(all_goals)}

=== SKILL GAPS (ranked by demand) ===
{skill_context if skill_context else "No significant skill gaps detected — candidate profile is strong."}

=== TOP INTERNSHIP OPPORTUNITIES (match score > 70%) ===
{chr(10).join(f"- {j}" for j in top_jobs) if top_jobs else "No high-match opportunities yet — suggest improving profile."}

=== PORTFOLIO STATUS ===
{portfolio_strength}
{chr(10).join(f"- {r}" for r in portfolio_recs) if portfolio_recs else "- Portfolio is strong, keep adding projects"}

=== INTERVIEW PRACTICE STATUS ===
{practice_label}
{f"Recommendation: {practice_rec}" if practice_rec else "On track — maintain consistent practice"}

=== INSTRUCTIONS ===
Generate EXACTLY 5 to 7 specific, actionable weekly steps ranked by impact.
Each step must be:
- Concrete and specific (not generic advice)
- Achievable within 1 week
- Directly tied to the candidate's gaps and opportunities above

Respond in this exact JSON format:
{{
  "actions": [
    "Step 1 text",
    "Step 2 text",
    "Step 3 text",
    "Step 4 text",
    "Step 5 text"
  ],
  "coaching_note": "One sentence motivational note for the candidate"
}}"""

    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are an expert AI career coach. Generate precise, actionable career plans in valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=600,
        )
        content = resp.choices[0].message.content.strip()
        result = json.loads(content)
        actions = result.get("actions", [])
        coaching_note = result.get("coaching_note", "")
        if coaching_note:
            actions.append(f"💬 Coach Note: {coaching_note}")
        logger.info("CareerStrategyAgent: LLM generated %d actions.", len(actions))
        return [str(a) for a in actions if a][:8]
    except Exception as exc:
        logger.error("CareerStrategyAgent: LLM generation failed - %s", exc)
        # Structured fallback
        fallback = []
        if top_skills:
            fallback.append(f"Dedicate 2 hours daily to learning {top_skills[0]} — it's needed in the most roles")
        if top_jobs:
            fallback.append(f"Submit your application to: {top_jobs[0]}")
        if portfolio_recs:
            fallback.extend(portfolio_recs[:2])
        if practice_rec:
            fallback.append(practice_rec)
        fallback.append("Research the tech stack of your target companies and note any new skill gaps")
        return fallback[:7]


# ==============================================================================
# MAIN FUNCTION — run_career_strategy_agent()
# ==============================================================================
def run_career_strategy_agent() -> dict:
    """
    Full pipeline:
      1. Load career goal
      2. Analyze skill gaps
      3. Analyze opportunity scores
      4. Analyze portfolio
      5. Analyze practice progress
      6. Generate LLM strategy
      7. Save to database/career_strategy.yaml
      8. Log activity
    Returns the strategy dict.
    """
    logger.info("CareerStrategyAgent: Starting comprehensive career analysis...")

    # Step 1
    primary_goal, all_goals = _load_career_goals()

    # Step 2
    top_skills, total_gap_count, skill_freq = _analyze_skill_gaps()

    # Step 3
    top_job_strings, top_job_dicts = _analyze_opportunities()

    # Step 4
    portfolio_num, portfolio_strength, portfolio_recs = _analyze_portfolio()

    # Step 5
    practice_count, practice_label, practice_rec = _analyze_practice()

    # Step 6
    actions = _generate_strategy(
        goal=primary_goal,
        all_goals=all_goals,
        top_skills=top_skills,
        skill_freq=skill_freq,
        top_jobs=top_job_strings,
        portfolio_num=portfolio_num,
        portfolio_strength=portfolio_strength,
        portfolio_recs=portfolio_recs,
        practice_count=practice_count,
        practice_label=practice_label,
        practice_rec=practice_rec,
    )

    # Step 7 — Save
    strategy_obj = {
        "strategy": {
            "goal": primary_goal,
            "all_goals": all_goals,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis": {
                "top_missing_skills": top_skills,
                "total_skill_gaps": total_gap_count,
                "top_opportunities": top_job_strings,
                "portfolio_strength": portfolio_strength,
                "portfolio_num_projects": portfolio_num,
                "practice_sessions": practice_count,
                "practice_status": practice_label,
            },
            "actions": actions,
        }
    }

    try:
        write_yaml_to_github(STRATEGY_FILE, strategy_obj)
        logger.info("CareerStrategyAgent: Strategy saved to %s", STRATEGY_FILE)
    except Exception as exc:
        logger.error("CareerStrategyAgent: Failed to save strategy - %s", exc)

    # Step 8 — Log
    try:
        append_log_entry({
            "agent": "CareerStrategyAgent",
            "action": f"Generated {len(actions)}-step weekly career strategy for goal: {primary_goal}",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception as exc:
        logger.warning("CareerStrategyAgent: Failed to log activity - %s", exc)

    logger.info("CareerStrategyAgent: Done. Goal='%s', Actions=%d", primary_goal, len(actions))
    return strategy_obj


# ==============================================================================
# Stand-alone entry point
# ==============================================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_career_strategy_agent()
    strategy = result.get("strategy", {})
    print(f"\n{'='*60}")
    print(f"🎯 CAREER STRATEGY — Goal: {strategy.get('goal')}")
    print(f"{'='*60}")
    print(f"\n📊 Analysis:")
    analysis = strategy.get("analysis", {})
    print(f"  Missing skills: {analysis.get('top_missing_skills')}")
    print(f"  Portfolio: {analysis.get('portfolio_strength')}")
    print(f"  Practice: {analysis.get('practice_status')}")
    print(f"\n📋 Weekly Action Plan:")
    for i, action in enumerate(strategy.get("actions", []), 1):
        print(f"  {i}. {action}")
    print(f"\nGenerated at: {strategy.get('generated_at')}")
