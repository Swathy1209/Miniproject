"""
repo_security_scanner_agent.py — Fast Repository Security Scanner (NO CLONING)
OrchestrAI Autonomous Multi-Agent System

Uses GitHub Code Search API + Content API to scan ALL repositories
for security vulnerabilities without cloning. Fast, cloud-compatible.
"""

from __future__ import annotations
import logging
import os
import re
import time
import requests
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

from backend.github_yaml_db import read_yaml_from_github, write_yaml_to_github, append_log_entry

load_dotenv()
logger = logging.getLogger("OrchestrAI.RepoSecurityScannerAgent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0) if GEMINI_API_KEY else None
from backend.utils.ai_engine import safe_llm_call as _safe_llm_call

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
USERS_FILE = "database/users.yaml"
SECURITY_REPORTS_FILE = "database/security_reports.yaml"

VULN_PATTERNS = [
    {"name": "Hardcoded Password",     "pattern": r'password\s*=\s*["\'][^"\']{4,}["\']',   "severity": "HIGH",   "rec": "Use os.getenv('PASSWORD') instead of hardcoded string"},
    {"name": "Hardcoded Secret Key",   "pattern": r'secret[_\s]?key\s*=\s*["\'][^"\']{4,}["\']', "severity": "HIGH",   "rec": "Load secret keys from environment variables"},
    {"name": "Hardcoded API Key",      "pattern": r'api[_\s]?key\s*=\s*["\'][^"\']{6,}["\']',     "severity": "HIGH",   "rec": "Store API keys in environment variables, never in code"},
    {"name": "AWS Secret Access Key",  "pattern": r'AWS_SECRET_ACCESS_KEY\s*=\s*["\'][^"\']+["\']', "severity": "HIGH",  "rec": "Use IAM roles or AWS Secrets Manager"},
    {"name": "Hardcoded Token",        "pattern": r'token\s*=\s*["\'][A-Za-z0-9_\-]{16,}["\']',   "severity": "HIGH",   "rec": "Store tokens in environment variables"},
    {"name": "Unsafe eval()",          "pattern": r'\beval\s*\(',                                    "severity": "HIGH",   "rec": "Avoid eval(); use json.loads() or ast.literal_eval() instead"},
    {"name": "Unsafe exec()",          "pattern": r'\bexec\s*\(',                                    "severity": "MEDIUM", "rec": "Avoid exec(); refactor to use functions directly"},
    {"name": "Pickle Deserialization", "pattern": r'pickle\.load\s*\(',                              "severity": "HIGH",   "rec": "Use JSON serialization instead of pickle"},
    {"name": "Subprocess Shell=True",  "pattern": r'subprocess\.\w+\s*\([^)]*shell\s*=\s*True',     "severity": "MEDIUM", "rec": "Pass argument list to subprocess; avoid shell=True"},
    {"name": "Unsafe YAML load",       "pattern": r'yaml\.load\s*\([^,)]+\)',                        "severity": "MEDIUM", "rec": "Use yaml.safe_load() instead of yaml.load()"},
    {"name": "SQL String Concat",      "pattern": r'["\'].*SELECT.+%s.*["\']|f["\'].*SELECT',        "severity": "HIGH",   "rec": "Use parameterized queries to prevent SQL injection"},
    {"name": "Debug Mode Enabled",     "pattern": r'DEBUG\s*=\s*True|debug\s*=\s*True',              "severity": "LOW",    "rec": "Disable debug mode in production"},
    {"name": "Weak Random",            "pattern": r'\brandom\.random\s*\(|\brandom\.randint\s*\(',   "severity": "LOW",    "rec": "Use secrets module for security-sensitive randomness"},
    {"name": "MD5 Hash Usage",         "pattern": r'hashlib\.md5\s*\(',                              "severity": "MEDIUM", "rec": "Use SHA-256 or stronger hashing algorithm"},
    {"name": "HTTP (not HTTPS)",       "pattern": r'http://(?!localhost|127\.0\.0\.1)',               "severity": "LOW",    "rec": "Use HTTPS endpoints to ensure encrypted communication"},
]

SEV_SCORE = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

def get_all_repos(username: str) -> list[dict]:
    """Fetch ALL public non-fork repos for the user."""
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}&sort=updated"
        try:
            resp = requests.get(url, headers=_gh_headers(), timeout=15)
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not batch:
                break
            repos.extend([r for r in batch if not r.get("fork") and not r.get("archived")])
            page += 1
        except Exception as exc:
            logger.error("Failed to fetch repos page %d: %s", page, exc)
            break
    logger.info("RepoSecurityScanner: Found %d repos for %s", len(repos), username)
    return repos

def get_python_files(owner: str, repo: str, branch: str = "main") -> list[dict]:
    """Get list of Python files in the repo using Git tree API."""
    for br in [branch, "master", "main"]:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{br}?recursive=1"
        try:
            resp = requests.get(url, headers=_gh_headers(), timeout=15)
            if resp.status_code == 200:
                tree = resp.json().get("tree", [])
                return [f for f in tree if f.get("path", "").endswith(".py") and f.get("type") == "blob"]
        except Exception:
            pass
    return []

def get_file_content(owner: str, repo: str, path: str) -> str:
    """Download raw file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""

def get_dependency_files(owner: str, repo: str) -> dict[str, str]:
    """Fetch dependency files to check for known vulnerable packages."""
    dep_files = {}
    for fname in ["requirements.txt", "package.json", "Pipfile", "setup.py"]:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{fname}"
        try:
            resp = requests.get(url, headers=_gh_headers(), timeout=8)
            if resp.status_code == 200:
                dep_files[fname] = resp.text[:3000]
        except Exception:
            pass
    return dep_files

def scan_content_for_vulns(content: str, file_path: str) -> list[dict]:
    """Apply regex heuristics to file content."""
    found = []
    lines = content.split("\n")
    for rule in VULN_PATTERNS:
        try:
            for i, line in enumerate(lines, 1):
                if re.search(rule["pattern"], line, re.IGNORECASE):
                    found.append({
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "file": file_path,
                        "line": i,
                        "snippet": line.strip()[:120],
                        "recommendation": rule["rec"],
                    })
                    break  # 1 incident per rule per file
        except Exception:
            pass
    return found

def _generate_fix(vuln: dict) -> str:
    """Use LLM to generate a specific code fix suggestion."""
    if not openai_client:
        return vuln.get("recommendation", "Apply secure coding practices.")
    prompt = (
        f"You are a cybersecurity engineer. Provide a 1-2 sentence fix for:\n"
        f"Vulnerability: {vuln['name']}\n"
        f"File: {vuln['file']}, Line {vuln['line']}\n"
        f"Code snippet: {vuln['snippet']}\n"
        f"Be specific and actionable."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120, temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return vuln.get("recommendation", "Apply secure coding practices.")

def scan_repo(owner: str, repo: dict) -> dict:
    """Fully scan one repo without cloning."""
    repo_name = repo.get("name", "")
    default_branch = repo.get("default_branch", "main")
    html_url = repo.get("html_url", f"https://github.com/{owner}/{repo_name}")
    
    logger.info("RepoSecurityScanner: Scanning %s/%s", owner, repo_name)
    
    all_vulns = []
    
    # Get Python files
    py_files = get_python_files(owner, repo_name, default_branch)
    logger.info("  %d Python files found", len(py_files))
    
    for f in py_files[:30]:  # Cap at 30 files per repo
        path = f.get("path", "")
        content = get_file_content(owner, repo_name, path)
        if content:
            found = scan_content_for_vulns(content, path)
            all_vulns.extend(found)
    
    # Check dependency files for vulnerable packages
    dep_files = get_dependency_files(owner, repo_name)
    RISKY_PACKAGES = ["flask==0.", "django==1.", "django==2.0", "pyyaml==5.3", "requests==2.18", "pillow==8."]
    for fname, content in dep_files.items():
        for pkg in RISKY_PACKAGES:
            if pkg.lower() in content.lower():
                all_vulns.append({
                    "name": f"Outdated/Vulnerable Package ({pkg})",
                    "severity": "MEDIUM",
                    "file": fname,
                    "line": 0,
                    "snippet": pkg,
                    "recommendation": f"Upgrade {pkg.split('=')[0]} to the latest secure version",
                })
    
    # Score
    score = sum(SEV_SCORE.get(v["severity"], 1) for v in all_vulns)
    if score == 0:
        risk_level = "Safe"
    elif score <= 3:
        risk_level = "Low"
    elif score <= 7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Generate AI fixes for top 3 issues
    top_vulns = sorted(all_vulns, key=lambda x: SEV_SCORE.get(x["severity"], 1), reverse=True)[:3]
    for v in top_vulns:
        v["fix"] = _generate_fix(v)
    
    formatted_issues = []
    for v in top_vulns:
        fix = v.get("fix", v.get("recommendation", ""))
        formatted_issues.append(
            f"[{v['severity']}] {v['name']} in `{v['file']}` line {v['line']} | Fix: {fix}"
        )
    
    return {
        "repo": repo_name,
        "repo_url": html_url,
        "risk_level": risk_level,
        "risk_score": score,
        "total_vulnerabilities": len(all_vulns),
        "vulnerabilities": [
            {"name": v["name"], "severity": v["severity"], "file": v["file"], "line": v["line"]}
            for v in all_vulns[:10]
        ],
        "issues": formatted_issues if formatted_issues else ["No critical issues detected."],
        "scanned_files": len(py_files),
        "auto_fix_pr": "",  # Will be filled by AutoFixPRGeneratorAgent
    }

def run_repo_security_scanner_agent() -> dict:
    logger.info("RepoSecurityScannerAgent: Starting FAST scan (no cloning)...")

    data = {}
    try:
        data = read_yaml_from_github(USERS_FILE) or {}
    except Exception:
        pass

    user = data.get("user", {})
    username = user.get("github_username", "Swathy1209")

    repos = get_all_repos(username)
    if not repos:
        logger.warning("RepoSecurityScannerAgent: No repos found.")
        return {}

    reports = []
    for repo in repos:
        try:
            report = scan_repo(username, repo)
            reports.append(report)
            time.sleep(0.5)  # Respect GitHub rate limits
        except Exception as exc:
            logger.error("RepoSecurityScannerAgent: Failed scanning %s - %s", repo.get("name"), exc)

    # Sort repos by risk score (highest first)
    _risk_order = {"High": 4, "Medium": 3, "Low": 2, "Safe": 1}
    reports.sort(key=lambda r: _risk_order.get(r.get("risk_level", "Safe"), 0), reverse=True)

    # Build priority_security_fix: single highest-severity issue across ALL repos
    priority_fix = {}
    best_score = -1
    for report in reports:
        repo_name = report.get("repo", "")
        repo_url = report.get("repo_url", "")
        risk_level = report.get("risk_level", "Safe")
        for vuln in report.get("vulnerabilities", []):
            sev_score = SEV_SCORE.get(vuln.get("severity", "LOW"), 1)
            if sev_score > best_score:
                best_score = sev_score
                fix_text = vuln.get("fix", vuln.get("recommendation", "Apply secure coding practices."))
                priority_fix = {
                    "repo": repo_name,
                    "repo_url": repo_url,
                    "risk": vuln.get("severity", ""),
                    "issue": vuln.get("name", ""),
                    "file": vuln.get("file", ""),
                    "line": vuln.get("line", 0),
                    "snippet": vuln.get("snippet", ""),
                    "fix": fix_text,
                }

    if not priority_fix and reports:
        priority_fix = {
            "repo": reports[0].get("repo", ""),
            "risk": "Safe",
            "issue": "No critical issues detected across all repositories.",
            "file": "", "line": 0, "fix": "All repositories passed security screening."
        }

    logger.info("RepoSecurityScannerAgent: Priority fix → %s in %s",
                priority_fix.get("issue"), priority_fix.get("repo"))

    payload = {
        "security_reports": reports,
        "priority_security_fix": priority_fix,
        "summary": {
            "total_repos": len(reports),
            "high_risk": sum(1 for r in reports if r.get("risk_level") == "High"),
            "medium_risk": sum(1 for r in reports if r.get("risk_level") == "Medium"),
            "low_risk": sum(1 for r in reports if r.get("risk_level") == "Low"),
            "safe": sum(1 for r in reports if r.get("risk_level") == "Safe"),
            "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }

    try:
        write_yaml_to_github(SECURITY_REPORTS_FILE, payload)
        append_log_entry({
            "agent": "RepoSecurityScannerAgent",
            "action": f"Scanned {len(reports)} repos, priority fix: {priority_fix.get('issue','')}",
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        })
    except Exception as exc:
        logger.error("RepoSecurityScannerAgent: Failed to save reports - %s", exc)

    risky = [r for r in reports if r["risk_level"] not in ("Safe",)]
    logger.info("RepoSecurityScannerAgent: Done. %d repos scanned, %d with issues.", len(reports), len(risky))
    return payload



if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    results = run_repo_security_scanner_agent()
    for r in results:
        print(f"\n{r['repo']}: {r['risk_level']} (score={r['risk_score']}, issues={r['total_vulnerabilities']})")
        for issue in r.get("issues", []):
            print(f"  {issue}")
