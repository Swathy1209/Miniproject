"""
auto_apply_agent.py — Auto Apply Preparation Agent
OrchestrAI Autonomous Multi-Agent System

Flow:
  1. Load database/jobs.yaml
  2. Load database/cover_letter_index.yaml
  3. Generate application package for each internship.
  4. Save resume copy to application_packages/resume.pdf on GitHub.
  5. Generate Markdown Application Package file.
  6. Save database/application_packages.yaml.
  7. Return application_packages to ExecutionAgent.
"""

from __future__ import annotations

import logging
import os
import re
import base64
import requests
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
    _auth_headers,
    GITHUB_BRANCH,
    GITHUB_USERNAME,
    GITHUB_REPO
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.AutoApplyAgent")

JOBS_FILE = "database/jobs.yaml"
COVER_LETTERS_FILE = "database/cover_letter_index.yaml"
APP_PACKAGES_DB_FILE = "database/application_packages.yaml"

def get_github_url(file_path: str) -> str:
    """Helper to cleanly build the final raw GitHub content URL."""
    repo_slug = GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
    return f"https://raw.githubusercontent.com/{repo_slug}/main/{file_path}"

def read_jobs() -> list[dict]:
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    except Exception as exc:
        logger.error("AutoApplyAgent: read_jobs failed - %s", exc)
        return []

def read_cover_letters() -> dict:
    try:
        data = read_yaml_from_github(COVER_LETTERS_FILE)
        cover_letters = data.get("cover_letters", [])
        if not isinstance(cover_letters, list):
            return {}
            
        cl_lookup = {
            (item.get("company", ""), item.get("role", "")): item.get("link", "")
            for item in cover_letters if isinstance(item, dict)
        }
        return cl_lookup
    except Exception as exc:
        logger.error("AutoApplyAgent: read_cover_letters failed - %s", exc)
        return {}

def copy_resume_to_app_packages(repo_path: str = "resumes/swathiga_resume.pdf") -> str:
    """Copies the master resume to application_packages directory."""
    new_path = "application_packages/resume.pdf"
    
    try:
        repo_slug = GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
        url = f"https://api.github.com/repos/{repo_slug}/contents/{repo_path}"
        resp = requests.get(url, headers=_auth_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
        
        if resp.status_code == 200:
            content_b64 = resp.json().get("content", "")
            if not content_b64:
                return ""
                
            # decode it back to bytes to pass correctly into the put_raw_file architecture
            # Alternatively we could just copy it using the github API, but let's use our built-in writer
            decoded_bytes = base64.b64decode(content_b64)
            # Since _put_raw_file requires text, we bypass it for binary upload.
            
            put_url = f"https://api.github.com/repos/{repo_slug}/contents/{new_path}"
            headers = _auth_headers()
            
            # check if exists to get sha
            get_resp = requests.get(put_url, headers=headers)
            sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
            
            payload = {
                "message": "feat: prepare application package resume",
                "content": content_b64, 
                "branch": GITHUB_BRANCH
            }
            if sha:
                payload["sha"] = sha
                
            put_resp = requests.put(put_url, headers=headers, json=payload, timeout=20)
            put_resp.raise_for_status()
            
            # return public raw link
            return get_github_url(new_path)
        else:
            logger.warning("AutoApplyAgent: Could not download resume from %s (Status: %d)", repo_path, resp.status_code)
            return ""
    except Exception as exc:
        logger.error("AutoApplyAgent: copy_resume_to_app_packages failed - %s", exc)
        return ""

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def create_markdown_package(pkg: dict) -> str:
    company = pkg["company"]
    role = pkg["role"]
    
    file_name = f"{_slugify(company)}_{_slugify(role)}.md"
    file_path = f"application_packages/{file_name}"
    
    content = f"""# Application Package

Company: {company}
Role: {role}

Resume:
{pkg['resume_link']}

Cover Letter:
{pkg['cover_letter_link']}

Apply Here:
{pkg['apply_link']}

Status:
{pkg['status']}
"""
    try:
        _, sha = _get_raw_file(file_path)
        ts = datetime.now(timezone.utc).isoformat()
        _put_raw_file(
            file_path,
            content,
            sha,
            f"feat: generate application package for {company} {role} — {ts}"
        )
        return get_github_url(file_path)
    except Exception as exc:
        logger.error("AutoApplyAgent: create_markdown_package failed - %s", exc)
        return ""

def update_app_packages_yaml(data: list[dict]) -> bool:
    try:
        return write_yaml_to_github(APP_PACKAGES_DB_FILE, data)
    except Exception as exc:
        logger.error("AutoApplyAgent: update_app_packages_yaml failed - %s", exc)
        return False

def log_agent_activity(action: str, status: str = "success") -> None:
    try:
        append_log_entry({
            "agent": "AutoApplyAgent",
            "action": action,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except:
        pass

def run_auto_apply_agent() -> list[dict]:
    logger.info("AutoApplyAgent: Starting...")
    
    jobs = read_jobs()
    cover_letters = read_cover_letters()
    
    if not jobs:
        logger.warning("AutoApplyAgent: Missing jobs.")
        log_agent_activity("Skipped package generation - missing data", "partial")
        return []
        
    master_resume_link = copy_resume_to_app_packages()

    application_packages = []
    
    for job in jobs:
        company = job.get("company", "Unknown")
        role = job.get("role", "Unknown")
        apply_link = job.get("apply_link", "#")
        
        # Link mapping
        cl_link = cover_letters.get((company, role), "#")
        
        pkg = {
            "company": company,
            "role": role,
            "resume_link": master_resume_link if master_resume_link else "Resume extraction failed",
            "cover_letter_link": cl_link if cl_link != "#" else "No Cover Letter Generated",
            "apply_link": apply_link,
            "status": "Ready to Apply"
        }
        
        # Generate markdown file
        package_link = create_markdown_package(pkg)
        pkg["application_package_link"] = package_link
        
        application_packages.append(pkg)
                
    if application_packages:
        update_app_packages_yaml(application_packages)
        log_agent_activity(f"Generated Application Packages for {len(application_packages)} internships")
        
    logger.info("AutoApplyAgent: Completed successfully.")
    return application_packages

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_auto_apply_agent()
    print("AutoApplyAgent finished generated index:", len(result))
