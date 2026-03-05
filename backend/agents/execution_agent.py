"""
execution_agent.py — The Final Orchestrator
OrchestrAI Autonomous Multi-Agent System
"""

import logging
import os
import smtplib
import json
import requests
from datetime import datetime, timezone
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()

from backend.agents.career_agent import run_career_agent
from backend.agents.interview_feedback_agent import run_interview_feedback_agent
from backend.agents.skill_agent import run_skill_agent
from backend.agents.cover_letter_agent import run_cover_letter_agent
from backend.agents.practice_agent import run_practice_agent
from backend.agents.resume_optimization_agent import run_resume_optimization_agent
from backend.agents.portfolio_builder_agent import run_portfolio_builder_agent
from backend.agents.per_internship_portfolio_agent import run_per_internship_portfolio_agent
from backend.agents.repo_security_scanner_agent import run_repo_security_scanner_agent
from backend.agents.auto_fix_pr_agent import run_auto_fix_pr_agent
from backend.agents.career_strategy_agent import run_career_strategy_agent
from backend.agents.career_readiness_agent import run_career_readiness_agent
from backend.agents.career_analytics_agent import run_career_analytics_agent
from backend.agents.interview_coach_agent import run_interview_coach_agent
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
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

def _send_via_resend(subject: str, html_content: str) -> bool:
    """Send email via Resend HTTP API (works on Render free tier)."""
    if not RESEND_API_KEY:
        return False
    try:
        payload = {
            "from": "OrchestrAI <onboarding@resend.dev>",
            "to": [EMAIL_RECEIVER],
            "subject": subject,
            "html": html_content,
        }
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        if resp.status_code in (200, 201):
            logger.info("ExecutionAgent: Email sent via Resend API to %s", EMAIL_RECEIVER)
            return True
        else:
            logger.error("ExecutionAgent: Resend API error %s - %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        logger.error("ExecutionAgent: Resend send failed - %s", e)
        return False

def _send_via_smtp(subject: str, html_content: str) -> bool:
    """Send email via SMTP (works locally, may be blocked on Render free tier)."""
    if not EMAIL_USER or not EMAIL_PASS:
        logger.error("ExecutionAgent: EMAIL_USER or EMAIL_PASS not configured")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"OrchestrAI <{EMAIL_USER}>"
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("Your email client does not support HTML. Please view it in a modern client.")
        msg.add_alternative(html_content, subtype="html")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logger.info("ExecutionAgent: Email sent via SMTP to %s", EMAIL_RECEIVER)
        return True
    except Exception as e:
        logger.error("ExecutionAgent: SMTP send failed - %s", e)
        return False

def send_email(subject: str, html_content: str) -> bool:
    """Send HTML email. Tries Resend HTTP API first (Render-compatible), falls back to SMTP."""
    if RESEND_API_KEY:
        if _send_via_resend(subject, html_content):
            return True
        logger.warning("ExecutionAgent: Resend failed, falling back to SMTP...")
    return _send_via_smtp(subject, html_content)

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

    base_url      = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    analytics_url = f"{base_url}/analytics"

    # STEP 1: Fetch internships
    run_career_agent()

    # STEP 1.5: Process any stored interview feedback → update skill gaps BEFORE analysis
    run_interview_feedback_agent()

    # STEP 2: Generate per-job skill gap analysis
    run_skill_agent()

    # STEP 2.45: Scan ALL Repositories for Security Vulnerabilities (no cloning)
    run_repo_security_scanner_agent()

    # STEP 2.46: Auto-generate security fix PRs for risky repos
    run_auto_fix_pr_agent()

    # STEP 2.47: Generate Portfolio Website
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

    # STEP 2.9: Generate Career Strategy
    run_career_strategy_agent()

    # STEP 2.92: Compute Career Readiness Score (uses security + skills + portfolio + practice)
    run_career_readiness_agent()

    # STEP 2.93: Generate Career Analytics Dashboard (Plotly HTML)
    try:
        analytics_url = run_career_analytics_agent() or analytics_url
    except Exception as exc:
        logger.warning("CareerAnalyticsAgent failed: %s", exc)

    # STEP 2.94: Generate per-internship mock interview pages
    run_interview_coach_agent()

    # STEP 2.95: Generate per-internship customized portfolio pages
    run_per_internship_portfolio_agent()

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
    strategy_data = read_yaml_from_github("database/career_strategy.yaml")
    readiness_data = read_yaml_from_github("database/career_readiness.yaml")
    interview_data = read_yaml_from_github("database/interview_sessions.yaml")
    per_internship_portfolio_data = read_yaml_from_github("database/per_internship_portfolios.yaml")

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

    per_internship_list = per_internship_portfolio_data.get("per_internship_portfolios", []) if isinstance(per_internship_portfolio_data, dict) else []
    per_internship_lookup = {
        (item.get("company", ""), item.get("role", "")): item.get("portfolio_url", "")
        for item in per_internship_list if isinstance(item, dict)
    }

    interview_list = interview_data.get("interview_sessions", []) if isinstance(interview_data, dict) else []
    interview_lookup = {
        (item.get("company", ""), item.get("role", "")): item.get("interview_link", "")
        for item in interview_list if isinstance(item, dict)
    }

    portfolio_url = portfolio_data.get("portfolio", {}).get("url", "#") if isinstance(portfolio_data, dict) else "#"
    portfolio_html = f'<a href="{portfolio_url}" style="background:#2e7d32;color:white;padding:8px 14px;border-radius:6px;text-decoration:none;display:inline-block;font-weight:600;min-width:max-content;">View Portfolio</a>' if portfolio_url != "#" else "Not Generated"

    security_reports = security_data.get("security_reports", []) if isinstance(security_data, dict) else []

    # Build per-repo security lookup
    sec_report_lookup = {}
    sec_insights_html = ""
    if security_reports:
        for report in security_reports:
            repo_name = report.get("repo", "Unknown")
            repo_url = report.get("repo_url", f"https://github.com/Swathy1209/{repo_name}")
            # Use risk_level field from new scanner, fallback to score-based
            risk_level = report.get("risk_level", "")
            if not risk_level:
                rs = report.get("risk_score", 0)
                risk_level = "High" if rs > 7 else "Medium" if rs > 3 else "Low" if rs > 0 else "Safe"
            risk_color = {"High": "red", "Medium": "orange", "Low": "#f9a825", "Safe": "green"}.get(risk_level, "gray")
            issues = report.get("issues", [])
            top_issue = str(issues[0]) if issues else "No issues found."
            pr_url = report.get("auto_fix_pr", "")
            total_vulns = report.get("total_vulnerabilities", 0)
            scanned_files = report.get("scanned_files", 0)

            sec_report_lookup[repo_name] = {"level": risk_level, "color": risk_color}

            pr_html = f' <a href="{pr_url}" style="background:#1565c0;color:white;padding:2px 8px;border-radius:3px;text-decoration:none;font-size:11px">View PR →</a>' if pr_url else ""
            sec_insights_html += (
                f'<li style="margin-bottom:10px">'
                f'<a href="{repo_url}" style="font-weight:bold;color:#1a237e">{repo_name}</a> — '
                f'Risk: <span style="color:{risk_color};font-weight:bold">{risk_level}</span>'
                f' | {total_vulns} vulns | {scanned_files} files scanned{pr_html}'
                f'<br><span style="font-size:12px;color:#555;margin-left:10px">⤷ {top_issue[:100]}</span>'
                f'</li>'
            )
    else:
        sec_insights_html = "<li>No security scans performed yet. Scanner runs automatically each day.</li>"

    # Overall security summary for email table column
    if sec_report_lookup:
        levels = [v["level"] for v in sec_report_lookup.values()]
        overall_level = "High" if "High" in levels else "Medium" if "Medium" in levels else "Low" if "Low" in levels else "Safe"
        overall_color = {"High": "red", "Medium": "orange", "Low": "#f9a825", "Safe": "green"}.get(overall_level, "gray")
        total_repos = len(sec_report_lookup)
        high_count = sum(1 for v in sec_report_lookup.values() if v["level"] == "High")
        med_count = sum(1 for v in sec_report_lookup.values() if v["level"] == "Medium")
        overall_sec_html = (
            f'<span style="background:{overall_color};color:white;padding:3px 8px;border-radius:4px;font-size:12px;font-weight:600">{overall_level}</span>'
            f'<br><span style="font-size:11px;color:#555">{total_repos} repos scanned<br>'
            f'{high_count} High &middot; {med_count} Medium</span>'
        )
    else:
        overall_sec_html = '<span style="color:#999;font-size:12px">Not Scanned</span>'
        
    strategy = strategy_data.get("strategy", {}) if isinstance(strategy_data, dict) else {}
    strategy_goal = strategy.get("goal", "Data Engineering Internship")
    strategy_actions = strategy.get("actions", [])
    strategy_analysis = strategy.get("analysis", {})

    # Build rich strategy HTML for email
    if strategy_actions:
        strategy_action_html = "".join(
            f'<li style="margin:6px 0;padding:4px 8px;background:#e8f5e9;border-left:3px solid #2e7d32;border-radius:3px">'
            f'{action}</li>'
            for action in strategy_actions
        )
    else:
        strategy_action_html = "<li>Keep practicing and building projects!</li>"

    # Analysis summary badges
    top_skills = strategy_analysis.get("top_missing_skills", [])
    portfolio_str = strategy_analysis.get("portfolio_strength", "")
    practice_str = strategy_analysis.get("practice_status", "")
    top_opps = strategy_analysis.get("top_opportunities", [])

    skill_badges = "".join(
        f'<span style="background:#ffebee;color:#c62828;padding:3px 8px;border-radius:12px;font-size:11px;margin:2px;display:inline-block">{s}</span>'
        for s in top_skills[:5]
    ) if top_skills else '<span style="color:green">✓ No critical skill gaps</span>'

    opp_list = "".join(f"<li style='font-size:12px;margin:3px 0'>{o}</li>" for o in top_opps[:3]) if top_opps else "<li>Run pipeline to identify top matches</li>"

    # Career Readiness Score badge
    cr = readiness_data.get("career_readiness", {}) if isinstance(readiness_data, dict) else {}
    readiness_score = cr.get("readiness_score", 0)
    readiness_label = cr.get("label", "")
    readiness_color = (
        "#2e7d32" if readiness_score >= 85 else
        "#1565c0" if readiness_score >= 70 else
        "#e65100" if readiness_score >= 50 else "#c62828"
    )
    readiness_html = (
        f'<div style="background:{readiness_color};color:white;border-radius:10px;padding:14px 20px;'
        f'display:inline-block;margin-bottom:16px">'
        f'<span style="font-size:28px;font-weight:700">{readiness_score}</span>'
        f'<span style="font-size:14px">/100</span>&nbsp;&nbsp;'
        f'<span style="font-size:15px;font-weight:600">{readiness_label}</span></div>'
    ) if readiness_score else '<span style="color:#999">Readiness score computing...</span>'

    # Priority Security Fix banner
    pf = security_data.get("priority_security_fix", {}) if isinstance(security_data, dict) else {}
    if pf and pf.get("issue") and pf.get("risk") in ("HIGH", "MEDIUM", "HIGH"):
        pf_risk_color = "red" if pf.get("risk") == "HIGH" else "orange"
        pf_repo_url = pf.get("repo_url", f"https://github.com/Swathy1209/{pf.get('repo','')}")
        priority_fix_html = (
            f'<div style="background:#fff3e0;border:2px solid #e65100;border-radius:10px;padding:16px;margin-bottom:20px">'
            f'<h3 style="color:#e65100;margin:0 0 10px 0">🚨 Priority Security Fix Required</h3>'
            f'<table style="width:100%;font-size:13px;border-collapse:collapse">'
            f'<tr><td style="color:#666;width:100px">Repository</td>'
            f'<td><a href="{pf_repo_url}" style="color:#1565c0;font-weight:600">{pf.get("repo","")}</a></td></tr>'
            f'<tr><td style="color:#666">Risk Level</td>'
            f'<td><span style="background:{pf_risk_color};color:white;padding:2px 8px;border-radius:3px;font-size:12px;font-weight:600">{pf.get("risk","")}</span></td></tr>'
            f'<tr><td style="color:#666">Vulnerability</td><td style="font-weight:600">{pf.get("issue","")}</td></tr>'
            f'<tr><td style="color:#666">File</td><td><code style="background:#f5f5f5;padding:2px 6px;border-radius:3px">{pf.get("file","")}:{pf.get("line","")}</code></td></tr>'
            f'<tr><td style="color:#666">Code</td><td><code style="background:#ffebee;padding:2px 6px;border-radius:3px;color:#b71c1c">{str(pf.get("snippet",""))[:80]}</code></td></tr>'
            f'<tr><td style="color:#666">Fix</td><td style="color:#2e7d32">{pf.get("fix","")}</td></tr>'
            f'</table></div>'
        )
    else:
        priority_fix_html = '<div style="background:#e8f5e9;border-radius:8px;padding:12px;margin-bottom:16px;color:#2e7d32">✅ No critical security issues detected across all repositories!</div>'

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

        # Per-internship customized portfolio (separate column)
        pip_url = per_internship_lookup.get(key, "")
        if pip_url:
            custom_portfolio_html = f'<a href="{pip_url}" style="background:#1565c0;color:white;padding:8px 14px;border-radius:6px;text-decoration:none;display:inline-block;font-weight:600;font-size:12px">🎯 Custom Portfolio</a><br/><span style="font-size:10px;color:#888">Tailored for this role</span>'
        else:
            custom_portfolio_html = '<span style="color:#999;font-size:12px">Not Generated</span>'

        # Interview Sim column
        interview_url = interview_lookup.get(key, "")
        if interview_url:
            interview_html = f'<a href="{interview_url}" style="background:#7c3aed;color:white;padding:8px 14px;border-radius:6px;text-decoration:none;font-weight:600;display:inline-block;font-size:12px">🎤 Start Mock Interview</a>'
        else:
            interview_html = '<span style="color:#999;font-size:12px">Not Generated</span>'

        rows += f"""
        <tr>
            <td style='padding:8px;border:1px solid #ddd'>{job.get('company', '')}</td>
            <td style='padding:8px;border:1px solid #ddd'>{job.get('role', '')}</td>
            <td style='padding:8px;border:1px solid #ddd;font-size:12px;color:#555'>{job.get('location', '')}</td>
            <td style='padding:8px;border:1px solid #ddd;font-size:12px'>{', '.join(job.get('technical_skills', []))}</td>
            <td style='padding:8px;border:1px solid #ddd'><a href="{job.get('apply_link','#')}" style="font-weight:bold;color:#1565c0">Apply</a>{app_html}</td>
            <td style='padding:8px;border:1px solid #ddd'>{score_html}</td>
            <td style='padding:8px;border:1px solid #ddd;color:#c62828;font-size:12px'>{missing_skills or '<span style="color:green">✓ All covered</span>'}</td>
            <td style='padding:8px;border:1px solid #ddd;font-size:12px;color:#1565c0'>{roadmap or '—'}</td>
            <td style='padding:8px;border:1px solid #ddd'>{cl_html}<br><br>{opt_html}</td>
            <td style='padding:8px;border:1px solid #ddd;text-align:center'>{overall_sec_html}</td>
            <td style='padding:8px;border:1px solid #ddd;text-align:center'>{custom_portfolio_html}</td>
            <td style='padding:8px;border:1px solid #ddd;text-align:center'>{interview_html}</td>
        </tr>
        """

    # STEP 6: Generate full HTML email
    html = f"""
    <html>
    <head><style>
      body {{ font-family: Arial, sans-serif; font-size: 13px; background: #f8f9fa; margin: 0; padding: 20px; }}
      h2, h3 {{ color: #1a237e; }}
      table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
      th {{ background: #1a237e; color: white; padding: 10px 8px; text-align: left; font-size: 12px; white-space: nowrap; }}
      tr:nth-child(even) td {{ background: #f5f5f5; }}
    </style></head>
    <body>
        <h2>&#x1F916; Daily AI &amp; Data Science Internship Report</h2>

        <!-- Career Readiness Score -->
        <div style="background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:20px;display:flex;align-items:center;gap:20px;flex-wrap:wrap">
          <div>
            <p style="color:#666;font-size:12px;margin:0 0 6px 0;font-weight:600">&#x1F3AF; CAREER READINESS SCORE</p>
            {readiness_html}
          </div>
          <div style="flex:1;min-width:200px">
            <p style="font-size:12px;color:#555;margin:0">
              <b>Skill Coverage:</b> {cr.get('components', dict()).get('skill_coverage', dict()).get('score', 0):.0f}/100 &nbsp;|&nbsp;
              <b>Portfolio:</b> {cr.get('components', dict()).get('portfolio_strength', dict()).get('score', 0):.0f}/100 &nbsp;|&nbsp;
              <b>Practice:</b> {cr.get('components', dict()).get('interview_practice', dict()).get('score', 0):.0f}/100 &nbsp;|&nbsp;
              <b>Security:</b> {cr.get('components', dict()).get('security_health', dict()).get('score', 0):.0f}/100
            </p>
          </div>
        </div>

        <!-- Priority Security Fix -->
        {priority_fix_html}

        <table>
            <tr>
                <th>Company</th>
                <th>Role</th>
                <th>Location</th>
                <th>Required Skills</th>
                <th>Apply</th>
                <th>Match Score</th>
                <th>Skill Gap</th>
                <th>Learning Roadmap</th>
                <th>Generated Assets</th>
                <th>&#x1F512; Security Risk</th>
                <th>&#x1F3AF; Custom Portfolio</th>
                <th>&#x1F3A4; Interview Sim</th>
            </tr>
            {rows}
        </table>


        <h3>&#x1F9ED; Career Strategy Recommendation</h3>
        <div style="background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:20px">
          <p style="font-size:15px;margin:0 0 12px 0">&#x1F3AF; <b>Goal:</b> {strategy_goal}</p>

          <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px">
            <div style="flex:1;min-width:200px">
              <p style="font-weight:bold;color:#c62828;margin:0 0 6px 0">&#x26A0;&#xFE0F; Top Skill Gaps to Close:</p>
              <div>{skill_badges}</div>
            </div>
            <div style="flex:1;min-width:200px">
              <p style="font-weight:bold;color:#1565c0;margin:0 0 6px 0">&#x1F4BC; Portfolio:</p>
              <span style="font-size:12px;color:#555">{portfolio_str}</span>
              <br><p style="font-weight:bold;color:#2e7d32;margin:6px 0 4px 0">&#x1F3A4; Practice Status:</p>
              <span style="font-size:12px;color:#555">{practice_str}</span>
            </div>
          </div>

          <p style="font-weight:bold;color:#1565c0;margin:12px 0 6px 0">&#x1F31F; Top Matching Opportunities to Apply Now:</p>
          <ul style="margin:0;padding-left:20px">{opp_list}</ul>

          <p style="font-weight:bold;margin:16px 0 8px 0">&#x1F4CB; This Week's Action Plan:</p>
          <ol style="margin:0;padding-left:20px">{strategy_action_html}</ol>
        </div>

        <h3>&#x1F510; Security Insights &mdash; All GitHub Repos</h3>
        <ul style="line-height:1.8">{sec_insights_html}</ul>

        <!-- Footer -->
        <div style="background:#1a1a2e;border-radius:12px;padding:20px;text-align:center;margin-top:24px">
          <p style="color:#9ca3af;font-size:12px;margin-bottom:12px">OrchestrAI Autonomous Career Intelligence System</p>
          <a href="{analytics_url}" style="background:linear-gradient(135deg,#7c3aed,#4f46e5);color:white;
             padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;
             display:inline-block;margin:4px">📊 View Career Analytics Dashboard</a>
          <a href="{base_url}" style="background:#1f2937;color:#9ca3af;
             padding:12px 24px;border-radius:8px;text-decoration:none;font-size:13px;
             display:inline-block;margin:4px">🏠 OrchestrAI Dashboard</a>
        </div>
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
