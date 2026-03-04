"""
repo_security_scanner_agent.py — Repository Security Scanner Agent
OrchestrAI Autonomous Multi-Agent System
"""

import logging
import os
import shutil
import subprocess
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from backend.github_yaml_db import read_yaml_from_github, write_yaml_to_github, append_log_entry

load_dotenv()
logger = logging.getLogger("OrchestrAI.RepoSecurityScannerAgent")

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=GEMINI_BASE_URL,
) if OPENAI_API_KEY else None

USERS_FILE = "database/users.yaml"
SECURITY_REPORTS_FILE = "database/security_reports.yaml"
TEMP_DIR = "./temp_repos"

def get_github_repos(username: str) -> list[dict]:
    headers = {}
    github_token = os.getenv("GITHUB_TOKEN", "")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            repos = resp.json()
            # Top 4 latest repos to avoid taking too much execution time
            return [r for r in repos if not r.get("fork") and not r.get("archived") and r.get("size", 0) > 0][:4]
    except Exception as exc:
        logger.error(f"Failed to fetch repos: {exc}")
    return []

def clone_repo(clone_url: str, repo_name: str) -> str:
    path = os.path.join(TEMP_DIR, repo_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(TEMP_DIR, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", clone_url, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path

def run_bandit(repo_path: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["bandit", "-r", repo_path, "-f", "json", "-q"],
            capture_output=True,
            text=True
        )
        data = json.loads(result.stdout)
        return data.get("results", [])
    except Exception as exc:
        logger.warning(f"Bandit scan missed or error (maybe no pure python files): {exc}")
    return []

def get_nvd_cves(deps: list[str]) -> list[str]:
    # Mocking NVD API because NVD API 2.0 has stringent rate limits and frequent blocks.
    # Reaching out directly usually results in 403 or timeouts without an API key. 
    return []

def _generate_fix(issue_text: str) -> str:
    if not openai_client:
        return "Review code and implement secure practices."
    prompt = f"Suggest a very brief (1 sentence) secure coding fix for the following Python vulnerability:\n{issue_text}"
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Review code."

def analyze_repo(repo: dict) -> dict:
    repo_name = repo.get("name")
    clone_url = repo.get("clone_url")
    logger.info(f"Scanning repo: {repo_name}")
    
    repo_path = clone_repo(clone_url, repo_name)
    issues = run_bandit(repo_path)
    
    shutil.rmtree(repo_path, ignore_errors=True)
    
    score = 0
    parsed_issues = []
    
    for issue in issues[:4]: # Cap issues
        sev = issue.get("issue_severity", "LOW").upper()
        if sev == "HIGH": score += 3
        elif sev == "MEDIUM": score += 2
        else: score += 1
        
        issue_desc = issue.get("issue_text", "")
        fix = _generate_fix(issue_desc)
        parsed_issues.append(f"[{sev}] {issue_desc} (Fix: {fix})")
        
    return {
        "repo": repo_name,
        "risk_score": score,
        "issues": parsed_issues if parsed_issues else ["Secure"]
    }

def run_repo_security_scanner_agent():
    logger.info("Starting RepoSecurityScannerAgent")
    data = {"user": {}}
    try:
        data = read_yaml_from_github(USERS_FILE)
    except:
        pass
        
    user = data.get("user", {})
    username = user.get("github_username", "Swathy1209")
    
    repos = get_github_repos(username)
    reports = []
    
    for r in repos:
        rep_data = analyze_repo(r)
        reports.append(rep_data)
        
    try:
        write_yaml_to_github(SECURITY_REPORTS_FILE, {"security_reports": reports})
        append_log_entry({
            "agent": "RepoSecurityScannerAgent",
            "action": "Generated Security Reports",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except Exception as exc:
        logger.error(f"Failed to save security reports: {exc}")
        
    logger.info("RepoSecurityScannerAgent finished.")
    return reports

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    run_repo_security_scanner_agent()
