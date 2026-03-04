"""
execution_agent.py — The Final Orchestrator
OrchestrAI Autonomous Multi-Agent System
"""

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

from backend.agents.career_agent import run_career_agent
from backend.agents.skill_agent import run_skill_agent
from backend.agents.cover_letter_agent import run_cover_letter_agent
from backend.agents.practice_agent import run_practice_agent
from backend.agents.resume_optimization_agent import run_resume_optimization_agent
from backend.agents.portfolio_builder_agent import run_portfolio_builder_agent
from backend.agents.repo_security_scanner_agent import run_repo_security_scanner_agent
from backend.agents.auto_apply_agent import run_auto_apply_agent
from backend.agents.opportunity_matching_agent import run_opportunity_matching_agent
from backend.github_yaml_db import read_yaml_from_github, append_log_entry

logger = logging.getLogger("OrchestrAI.ExecutionAgent")

# Email Configuration
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))

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

def __log_activity(action: str):
    try:
        append_log_entry({
            "agent": "ExecutionAgent",
            "action": action,
            "status": "success",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except:
        pass

def run_orchestrai_pipeline():
    logging.info("Starting OrchestrAI pipeline")

    # STEP 1: Fetch internships
    run_career_agent()

    # STEP 2: Generate per-job skill gap analysis
    run_skill_agent()

    # STEP 2.45: Scan Repositories for Security Vulnerabilities
    run_repo_security_scanner_agent()

    # STEP 2.5: Generate Portfolio Website
    run_portfolio_builder_agent()

    # STEP 2.5: Generate cover letters
    run_cover_letter_agent()

    # STEP 2.55: Generate interview practice portals
    run_practice_agent()

    # STEP 2.6: Optimize Resumes
    run_resume_optimization_agent()

    # STEP 2.7: Generate Application Packages
    run_auto_apply_agent()

    # STEP 2.8: Compute Opportunity Matching
    run_opportunity_matching_agent()

    # STEP 3: Read GitHub database
    jobs_data = read_yaml_from_github("database/jobs.yaml")
    skill_gap_data = read_yaml_from_github("database/skill_gap_per_job.yaml")
    cover_letter_data = read_yaml_from_github("database/cover_letter_index.yaml")
    optimization_data = read_yaml_from_github("database/resume_optimizations.yaml")
    apply_packages_data = read_yaml_from_github("database/application_packages.yaml")
    scores_data = read_yaml_from_github("database/opportunity_scores.yaml")
    practice_data = read_yaml_from_github("database/practice_sessions.yaml")
    portfolio_data = read_yaml_from_github("database/portfolio.yaml")
    security_data = read_yaml_from_github("database/security_reports.yaml")

    jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []
    skill_analysis = skill_gap_data.get("job_skill_analysis", []) if isinstance(skill_gap_data, dict) else []

    # STEP 4: Convert skill & cover letter analysis to lookup dictionaries
    skill_lookup = {
        (item.get("company", ""), item.get("role", "")): item
        for item in skill_analysis if isinstance(item, dict)
    }
    
    cover_letters = cover_letter_data.get("cover_letters", []) if isinstance(cover_letter_data, dict) else []
    cl_lookup = {
        (item.get("company", ""), item.get("role", "")): item.get("link", "#")
        for item in cover_letters if isinstance(item, dict)
    }
    
    opt_records = optimization_data if isinstance(optimization_data, list) else []
    opt_lookup = {
        (item.get("company", ""), item.get("role", "")): item.get("optimized_resume_link", "#")
        for item in opt_records if isinstance(item, dict)
    }

    apply_packages = apply_packages_data if isinstance(apply_packages_data, list) else []
    app_pkg_lookup = {
        (item.get("company", ""), item.get("role", "")): item
        for item in apply_packages if isinstance(item, dict)
    }

    scores_list = scores_data if isinstance(scores_data, list) else []
    score_lookup = {
        (item.get("company", ""), item.get("role", "")): item
        for item in scores_list if isinstance(item, dict)
    }

    practice_list = practice_data if isinstance(practice_data, list) else []
    practice_lookup = {
        (item.get("company", ""), item.get("role", "")): item.get("practice_link", "")
        for item in practice_list if isinstance(item, dict)
    }

    portfolio_url = portfolio_data.get("portfolio", {}).get("url", "#") if isinstance(portfolio_data, dict) else "#"
    portfolio_html = f'<a href="{portfolio_url}" style="background:#2e7d32;color:white;padding:8px 14px;border-radius:6px;text-decoration:none;display:inline-block;font-weight:600;min-width:max-content;">View Portfolio</a>' if portfolio_url != "#" else "Not Generated"

    security_reports = security_data.get("security_reports", []) if isinstance(security_data, dict) else []
    sec_insights_html = ""
    if security_reports:
        for report in security_reports:
            repo_name = report.get("repo", "Unknown")
            risk_score = report.get("risk_score", 0)
            issues = report.get("issues", [])
            risk_level = "High" if risk_score > 4 else "Medium" if risk_score > 1 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            top_issue = str(issues[0]) if issues else "No issues found."
            sec_insights_html += f"<li><strong>{repo_name}</strong> - Risk Level: <span style='color:{risk_color};font-weight:bold;'>{risk_level}</span><br><i>Recommended Fix:</i> {top_issue}</li>"
    else:
        sec_insights_html = "<li>No security scans performed yet.</li>"

    # STEP 5: Generate HTML table rows
    rows = ""

    for job in jobs:
        if not isinstance(job, dict):
            continue
            
        key = (job.get("company", ""), job.get("role", ""))
        analysis = skill_lookup.get(key, {})

        missing_skills = ", ".join(analysis.get("missing_skills", []))
        roadmap = " &rarr; ".join(analysis.get("roadmap", []))
        
        cl_link = cl_lookup.get(key, "#")
        if cl_link and cl_link != "#":
            cl_html = f'<a href="{cl_link}" style="background:#2e7d32; color:white; padding:6px 12px; text-decoration:none; border-radius:6px; display:inline-block; margin-bottom: 4px;">Cover Letter</a>'
        else:
            cl_html = "Not Generated"
            
        opt_link = opt_lookup.get(key, "#")
        if opt_link and opt_link != "#":
            opt_html = f'<a href="{opt_link}" style="background:#0277bd; color:white; padding:6px 12px; text-decoration:none; border-radius:6px; display:inline-block;">Optimized Resume</a>'
        else:
            opt_html = "Not Generated"
            
        app_pkg = app_pkg_lookup.get(key, {})
        app_status = app_pkg.get("status", "Not Generated")
        app_link = app_pkg.get("application_package_link", "#")
        
        status_color = "#f29900" if app_status == "Not Generated" else "#1a73e8"
        
        if app_link and app_link != "#":
            app_html = f'<br><br><a href="{app_link}" style="padding:4px 8px; border-radius:4px; font-weight:bold; color:white; background-color:{status_color}; font-size:11px; text-decoration:none; display:inline-block;">{app_status}</a>'
        else:
            app_html = f'<br><br><span style="padding:4px 8px; border-radius:4px; font-weight:bold; color:white; background-color:{status_color}; font-size:11px; display:inline-block;">{app_status}</span>'

        score_info = score_lookup.get(key, {})
        match_score = score_info.get("match_score", 0)
        prob = score_info.get("selection_probability", "Unknown")
        priority = score_info.get("priority", "Unknown")
        
        prob_color = "#2e7d32" if prob == "High" else "#f29900" if prob == "Medium" else "#d32f2f"
        
        score_html = f"""
        <b>Score:</b> {match_score}/100<br>
        <span style="color:{prob_color}; font-weight:bold; font-size:12px;">{prob} Probability</span><br>
        <span style="font-size:11px; background:#f1f3f4; padding:2px 4px; border-radius:3px;">{priority}</span>
        """

        practice_link = practice_lookup.get(key, "")
        if practice_link:
            practice_html = f'<a href="{practice_link}" style="background:#1976d2; color:white; padding:8px 14px; border-radius:6px; text-decoration:none; display:inline-block; font-weight:600;">Start Practice</a>'
        else:
            practice_html = '<span style="color:#999;">Not Generated</span>'

        rows += f"""
        <tr>
            <td>{job.get('company', '')}</td>
            <td>{job.get('role', '')}</td>
            <td>{job.get('location', '')}</td>
            <td>{', '.join(job.get('technical_skills', []))}</td>
            <td><a href="{job.get('apply_link','#')}" style="font-weight:bold;">Apply</a>{app_html}</td>
            <td>{score_html}</td>
            <td>{missing_skills}</td>
            <td>{roadmap}</td>
            <td>{cl_html}<br><br>{opt_html}</td>
            <td>{practice_html}</td>
            <td>{portfolio_html}</td>
        </tr>
        """

    # STEP 6: Generate full HTML email
    html = f"""
    <html>
    <body>
        <h2>Daily AI & Data Science Internship Report</h2>

        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr style="background-color: #f1f3f4; text-align: left;">
                <th>Company</th>
                <th>Role</th>
                <th>Location</th>
                <th>Technical Skills</th>
                <th>Apply</th>
                <th>Match Score</th>
                <th>Skill Gap</th>
                <th>Learning Roadmap</th>
                <th>Generated Assets</th>
                <th>Practice Time</th>
                <th>Portfolio</th>
            </tr>
            {rows}
        </table>

        <h3>Security Insights</h3>
        <ul>
            {sec_insights_html}
        </ul>
    </body>
    </html>
    """

    # Removing the sleep timezone delay to ensure email sends immediately
    logging.info("Sending email immediately.")

    # Actually send email
    send_email(
        subject="Daily AI & Data Science Internship Report with Skill Gap Analysis",
        html_content=html
    )

    logging.info("OrchestrAI pipeline completed successfully")
    __log_activity("Pipeline execution completed and email sent")
