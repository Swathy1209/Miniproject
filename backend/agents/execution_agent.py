"""
execution_agent.py — The Final Orchestrator
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Reads database/jobs.yaml from GitHub
  2. Reads database/skill_gap_per_job.yaml from GitHub
  3. Consolidates everything into a clean HTML table email
  4. Sends the email using SMTP (e.g. Gmail)
  5. Logs activity to database/agent_logs.yaml
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

from backend.github_yaml_db import read_yaml_from_github, append_log_entry

logger = logging.getLogger("OrchestrAI.ExecutionAgent")

# File paths in GitHub repo
JOBS_FILE      = "database/jobs.yaml"
SKILL_GAP_FILE = "database/skill_gap_per_job.yaml"

# Email Configuration
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))


def read_database() -> tuple[list[dict], list[dict]]:
    """
    Read both jobs.yaml and skill_gap_per_job.yaml from GitHub database.
    Returns: (jobs_list, job_skill_analysis_list)
    """
    jobs = []
    job_skill_analysis = []

    try:
        jobs_data = read_yaml_from_github(JOBS_FILE)
        jobs = jobs_data.get("jobs", [])
        if not isinstance(jobs, list):
            jobs = []
        logger.info("ExecutionAgent: Fetched %d jobs from GitHub.", len(jobs))
    except Exception as e:
        logger.error("ExecutionAgent: Failed to read jobs.yaml - %s", e)

    try:
        gap_data = read_yaml_from_github(SKILL_GAP_FILE)
        job_skill_analysis = gap_data.get("job_skill_analysis", [])
        if not isinstance(job_skill_analysis, list):
            job_skill_analysis = []
        logger.info("ExecutionAgent: Fetched job_skill_analysis list.")
    except Exception as e:
        logger.error("ExecutionAgent: Failed to read skill_gap_per_job.yaml - %s", e)

    return jobs, job_skill_analysis


def generate_html_email(jobs: list[dict], job_skill_analysis: list[dict]) -> str:
    """Generate a responsive HTML email containing a table of jobs and skill gaps."""
    date_str = datetime.now().strftime("%d %B %Y")
    
    # Create a quick lookup for skill gaps: key = (company, role)
    gap_lookup = {}
    for analysis in job_skill_analysis:
        comp = analysis.get("company", "Unknown Company").lower()
        role = analysis.get("role", "Unknown Role").lower()
        gap_lookup[(comp, role)] = {
            "missing_skills": analysis.get("missing_skills", []),
            "roadmap": analysis.get("roadmap", [])
        }

    html = f"""
    <html>
      <body style="font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f4f6f9; color: #333; margin:0; padding:20px;">
        <div style="max-width: 1200px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
          
          <!-- HEADER -->
          <div style="background-color: #1a73e8; color: #ffffff; padding: 20px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px;">OrchestrAI Daily Briefing</h1>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">{date_str}</p>
          </div>
          
          <div style="padding: 20px;">
            <p>Hi there,</p>
            <p>Here is your daily consolidated report of new AI/Data Science internships, along with personalized skill gap analyses and learning roadmaps per job.</p>
    """

    if jobs:
        html += """
          <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; min-width: 900px; font-size: 13px;">
              <thead>
                <tr style="background-color: #f1f3f4; text-align: left;">
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Company</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Role</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Location</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Role Keywords</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Technical Skills</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Skill Gap</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Learning Roadmap</th>
                  <th style="padding: 12px; border-bottom: 2px solid #ddd;">Apply</th>
                </tr>
              </thead>
              <tbody>
        """

        for job in jobs:
            company = job.get("company", "Unknown Company")
            role = job.get("role", "Unknown Role")
            location = job.get("location", "Location Not Specified")
            link = job.get("apply_link", "#")
            req_skills = job.get("technical_skills", [])
            keywords = job.get("role_keywords", [])

            # Format skills/keywords
            kws_str = ", ".join(keywords) if keywords else "-"
            skills_str = ", ".join(req_skills) if req_skills else "-"
            
            # Fetch corresponding gap analysis
            gap_info = gap_lookup.get((company.lower(), role.lower()), {})
            missing_skills = gap_info.get("missing_skills", [])
            roadmap = gap_info.get("roadmap", [])
            
            missing_str = ", ".join(missing_skills) if missing_skills else "None"
            
            # Use arrows for roadmap steps
            if roadmap:
                # If the AI started the step with a dash or bullet, clean it up
                clean_roadmap = [step.lstrip("-•* ") for step in roadmap]
                roadmap_str = " &rarr; ".join(clean_roadmap)
            else:
                roadmap_str = "Ready to apply!"

            html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                  <td style="padding: 12px; vertical-align: top;"><strong>{company}</strong></td>
                  <td style="padding: 12px; vertical-align: top;">{role}</td>
                  <td style="padding: 12px; vertical-align: top;">{location}</td>
                  <td style="padding: 12px; vertical-align: top; color: #555;">{kws_str}</td>
                  <td style="padding: 12px; vertical-align: top; color: #555;">{skills_str}</td>
                  <td style="padding: 12px; vertical-align: top; color: #d93025; font-weight: bold;">{missing_str}</td>
                  <td style="padding: 12px; vertical-align: top; color: #188038;">{roadmap_str}</td>
                  <td style="padding: 12px; vertical-align: top;">
                    <a href="{link}" style="display:inline-block; background-color:#1a73e8; color:#fff; text-decoration:none; padding:6px 12px; border-radius:4px; font-weight:bold; white-space: nowrap;">Apply &rarr;</a>
                  </td>
                </tr>
            """
            
        html += """
              </tbody>
            </table>
          </div>
        """
    else:
        html += "<p>No new internship listings matched your criteria today.</p>"

    # -- Footer --
    html += """
          </div>
          <div style="background-color: #f1f3f4; color: #666; padding: 15px; text-align: center; font-size: 12px;">
            <p style="margin: 0;">Automated by <b>OrchestrAI Execution Agent</b></p>
          </div>
        </div>
      </body>
    </html>
    """
    
    return html


def send_email(subject: str, html_content: str) -> bool:
    """Send the HTML email using SMTP."""
    if not EMAIL_USER or not EMAIL_PASS:
        logger.error("ExecutionAgent: EMAIL_USER or EMAIL_PASS not configured in .env")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"OrchestrAI <{EMAIL_USER}>"
    msg["To"] = EMAIL_RECEIVER

    # Set content type to HTML
    msg.set_content("Your email client does not support HTML. Please view it in a modern client.")
    msg.add_alternative(html_content, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logger.info("ExecutionAgent: Successfully sent email to %s", EMAIL_RECEIVER)
        return True
    except Exception as e:
        logger.error("ExecutionAgent: Failed to send email - %s", e)
        return False


def log_agent_activity(action: str, details: str = None, status: str = "success") -> bool:
    """Append a log entry to database/agent_logs.yaml."""
    entry = {
        "agent": "ExecutionAgent",
        "action": action,
        "status": status,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if details:
        entry["details"] = details
    try:
        return append_log_entry(entry)
    except Exception as exc:
        logger.error("ExecutionAgent: log_agent_activity failed - %s", exc)
        return False


def run_execution_agent() -> dict:
    """
    Main pipeline for ExecutionAgent.
    1. Read jobs list and skill gaps.
    2. Format beautiful email.
    3. Send email.
    4. Log actions.
    """
    logger.info("ExecutionAgent: Starting process...")
    log_agent_activity("Consolidating daily report and sending email")

    jobs, job_skill_analysis = read_database()

    subject = f"🚀 OrchestrAI: {len(jobs)} Internships & Per-Job AI Roadmaps"
    html_content = generate_html_email(jobs, job_skill_analysis)

    sent_ok = send_email(subject, html_content)
    
    if sent_ok:
        log_agent_activity("Daily email sent successfully", f"Sent to {EMAIL_RECEIVER}")
        return {"status": "success", "jobs_included": len(jobs), "recipient": EMAIL_RECEIVER}
    else:
        log_agent_activity("Email delivery failed", status="error")
        return {"status": "error"}


if __name__ == "__main__":
    import json, sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_execution_agent()
    print("\n--- ExecutionAgent Result ---")
    print(json.dumps(result, indent=2, preserve_tuple=True))
