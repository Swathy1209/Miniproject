"""
execution_agent.py — The Final Orchestrator
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Reads database/jobs.yaml from GitHub
  2. Reads database/skill_gap.yaml from GitHub
  3. Consolidates everything into a clean, modern HTML email
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
SKILL_GAP_FILE = "database/skill_gap.yaml"

# Email Configuration
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))


def read_database() -> tuple[list[dict], dict]:
    """
    Read both jobs.yaml and skill_gap.yaml from GitHub database.
    Returns: (jobs_list, skill_gap_dict)
    """
    jobs = []
    skill_gap = {}

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
        skill_gap = gap_data.get("skill_analysis", {})
        logger.info("ExecutionAgent: Fetched skill_gap analysis.")
    except Exception as e:
        logger.error("ExecutionAgent: Failed to read skill_gap.yaml - %s", e)

    return jobs, skill_gap


def generate_html_email(jobs: list[dict], skill_gap: dict) -> str:
    """Generate a clean, modern HTML email wrapping all the data."""
    date_str = datetime.now().strftime("%d %B %Y")
    
    # Extract skill gap data
    current_skills = skill_gap.get("current_skills", [])
    missing_skills = skill_gap.get("missing_skills", [])
    roadmap = skill_gap.get("recommended_learning_roadmap", [])
    user_name = skill_gap.get("user", "User").title()

    # -- HTML Styling Constants --
    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; color: #333; margin:0; padding:20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
          
          <!-- HEADER -->
          <div style="background-color: #1a73e8; color: #ffffff; padding: 20px; text-align: center;">
            <h1 style="margin: 0; font-size: 24px;">OrchestrAI Daily Briefing</h1>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">{date_str}</p>
          </div>
          
          <div style="padding: 20px;">
            <p>Hi <b>{user_name}</b>,</p>
            <p>Here is your daily consolidated report of new AI/Data Science internships and your personalized learning roadmap.</p>
    """

    # -- Skill Gap Section --
    html += """
            <h2 style="color: #1a73e8; border-bottom: 2px solid #e8eaed; padding-bottom: 5px; margin-top: 30px;">
              🧠 Your Learning Roadmap
            </h2>
    """
    if missing_skills and roadmap:
        html += f"""
            <p><b>Target Skills to Learn:</b> {', '.join(missing_skills)}</p>
            <div style="background-color: #fef7e0; border-left: 4px solid #fbbc04; padding: 10px 15px; margin-bottom: 20px; border-radius: 4px;">
              <ul style="margin: 0; padding-left: 20px;">
        """
        for step in roadmap:
            html += f"<li style='margin-bottom: 5px;'>{step}</li>"
        html += """
              </ul>
            </div>
        """
    else:
        html += "<p>You are fully equipped with all the required skills for today's jobs! 🎉</p>"

    # -- Jobs Section --
    html += f"""
            <h2 style="color: #1a73e8; border-bottom: 2px solid #e8eaed; padding-bottom: 5px; margin-top: 30px;">
              💼 Latest Internships ({len(jobs)})
            </h2>
    """
    if jobs:
        for job in jobs:
            title = job.get("role", "Unknown Role")
            company = job.get("company", "Unknown Company")
            link = job.get("apply_link", "#")
            req_skills = job.get("technical_skills", [])
            location = job.get("location", "Location Not Specified")

            skill_tags = "".join(
                [f"<span style='display:inline-block; background:#e8f0fe; color:#1a73e8; font-size:12px; padding:3px 8px; border-radius:12px; margin:2px;'>{s}</span>" for s in req_skills]
            )

            html += f"""
            <div style="border: 1px solid #e8eaed; border-radius: 6px; padding: 15px; margin-bottom: 15px;">
              <h3 style="margin:0 0 5px 0; font-size:18px;">{title} <span style="font-weight:normal; color:#666;">at</span> {company}</h3>
              <p style="margin:0 0 10px 0; font-size:14px; color:#555;">📍 {location}</p>
              <div style="margin-bottom:12px;">{skill_tags}</div>
              <a href="{link}" style="display:inline-block; background-color:#1a73e8; color:#fff; text-decoration:none; padding:8px 15px; border-radius:4px; font-size:14px; font-weight:bold;">Apply Now &rarr;</a>
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

    jobs, skill_gap = read_database()

    # Even if they are empty, we might still want to send a status email
    subject = f"🚀 OrchestrAI Daily Briefing: {len(jobs)} Internships & Your Roadmap"
    html_content = generate_html_email(jobs, skill_gap)

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
