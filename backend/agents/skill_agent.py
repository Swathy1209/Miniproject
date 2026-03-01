"""
skill_agent.py — Skill Gap Analysis Agent
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Read resume skills from database/users.yaml
  2. Read internships from database/jobs.yaml
  3. Generate skill gap PER JOB
  4. Generate learning roadmap PER JOB using OpenAI
  5. Store result in database/skill_gap_per_job.yaml
  6. Log agent activity to database/agent_logs.yaml
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
)
from backend.utils.ai_engine import generate_per_job_roadmap

logger = logging.getLogger("OrchestrAI.SkillAgent")

USERS_FILE     = "database/users.yaml"
JOBS_FILE      = "database/jobs.yaml"
SKILL_GAP_PER_JOB_FILE = "database/skill_gap_per_job.yaml"


# ==============================================================================
# Helper functions for database interaction
# ==============================================================================

def read_user_skills_yaml() -> list[str]:
    """
    Read extracted skills from database/users.yaml on GitHub.
    Returns: list of skills.
    """
    try:
        data = read_yaml_from_github(USERS_FILE)
        skills = data.get("user", {}).get("resume_skills", [])
        if not isinstance(skills, list):
            skills = []
        logger.info("SkillAgent: Found %d skills in users.yaml.", len(skills))
        return [str(s).strip() for s in skills if s]
    except Exception as exc:
        logger.error("SkillAgent: read_user_skills_yaml failed - %s", exc)
        return []

def read_jobs_yaml() -> list[dict]:
    """Read jobs.yaml from GitHub."""
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list):
            return []
        logger.info("SkillAgent: Read %d jobs from '%s'.", len(jobs), JOBS_FILE)
        return jobs
    except Exception as exc:
        logger.error("SkillAgent: read_jobs_yaml failed - %s", exc)
        return []


def store_skill_gap_per_job_yaml(job_skill_analysis: list[dict]) -> bool:
    """Write skill gaps and roadmap per job to database/skill_gap_per_job.yaml on GitHub."""
    data = {
        "job_skill_analysis": job_skill_analysis
    }
    try:
        ok = write_yaml_to_github(SKILL_GAP_PER_JOB_FILE, data)
        if ok:
            logger.info(
                "SkillAgent: %s written (%d jobs).", 
                SKILL_GAP_PER_JOB_FILE, len(job_skill_analysis)
            )
        return ok
    except Exception as exc:
        logger.error("SkillAgent: store_skill_gap_per_job_yaml failed - %s", exc)
        return False


def log_agent_activity(action: str, details: Optional[str] = None, status: str = "success") -> bool:
    """Append a log entry to database/agent_logs.yaml."""
    entry = {
        "agent": "SkillAgent",
        "action": action,
        "status": status,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if details:
        entry["details"] = details
    try:
        return append_log_entry(entry)
    except Exception as exc:
        logger.error("SkillAgent: log_agent_activity failed - %s", exc)
        return False


# ==============================================================================
# Main Orchestrator (run_skill_agent)
# ==============================================================================

def run_skill_agent() -> dict:
    """
    Execute full SkillAgent pipeline.
    Does NOT send email! Outputs JSON format needed by ExecutionAgent.
    """
    logger.info("SkillAgent: Starting Skill Agent...")
    log_agent_activity("SkillAgent run initiated")
    
    result = {
        "job_skill_analysis": [],
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "error"
    }
    
    try:
        # Step 1: Read resume skills 
        resume_skills = read_user_skills_yaml()
        if not resume_skills:
            logger.warning("SkillAgent: No skills extracted from user file.")
            log_agent_activity("Skill analysis skipped - no resume skills", status="partial")
            result["status"] = "partial"
            return result
            
        user_lower = {s.lower() for s in resume_skills}

        # Step 2: Read jobs
        jobs = read_jobs_yaml()
        if not jobs:
            log_agent_activity("Skill analysis skipped - no jobs found in database", status="partial")
            result["status"] = "partial"
            return result

        # Steps 3 & 4: Generate skill gap and roadmap PER JOB
        job_skill_analysis = []
        
        for job in jobs:
            company = job.get("company", "Unknown Company")
            role = job.get("role", "Unknown Role")
            tech_skills = job.get("technical_skills", [])
            
            # Compute missing skills for this job
            job_skill_set: set[str] = set()
            for s in tech_skills:
                if s and str(s).strip():
                    job_skill_set.add(str(s).strip())
                    
            missing = sorted({s for s in job_skill_set if s.lower() not in user_lower})
            job_skills_list = sorted(job_skill_set)
            
            # Generate learning roadmap for this job
            roadmap = generate_per_job_roadmap(resume_skills, job_skills_list, missing)
            
            job_analysis = {
                "company": company,
                "role": role,
                "missing_skills": missing,
                "roadmap": roadmap
            }
            job_skill_analysis.append(job_analysis)

        # Step 5: Store skill gap data per job
        if not store_skill_gap_per_job_yaml(job_skill_analysis):
            logger.error("SkillAgent: Failed writing skill_gap_per_job.yaml.")
            result["status"] = "error"
            return result

        # Step 6: Log success
        log_agent_activity("Skill gap analysis complete", f"Analyzed {len(jobs)} jobs.")
        
        # Step 7: Return structured output
        result["job_skill_analysis"] = job_skill_analysis
        result["status"] = "success"
        result["analyzed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        logger.info("SkillAgent: Completed successfully.")
        
        return result

    except Exception as exc:
        logger.exception("SkillAgent: Pipeline crashed - %s", exc)
        log_agent_activity("Exception in SkillAgent", str(exc), "error")
        result["status"] = "error"
        return result


if __name__ == "__main__":
    import json, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    res = run_skill_agent()
    print("\n--- SkillAgent Output ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))
