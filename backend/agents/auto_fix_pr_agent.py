"""
auto_fix_pr_agent.py — Automated Security Fix Pull Request Generator
OrchestrAI Autonomous Multi-Agent System

Reads security scan results, generates unified diffs for each vulnerability,
creates a branch, commits the fix, and opens a Pull Request on GitHub.
"""

from __future__ import annotations
import base64
import json
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
logger = logging.getLogger("OrchestrAI.AutoFixPRGeneratorAgent")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0) if GEMINI_API_KEY else None

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
SECURITY_REPORTS_FILE = "database/security_reports.yaml"

def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

# ── GitHub API helpers ─────────────────────────────────────────────────────────

def _get_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
    except Exception:
        pass
    return "main"

def _get_branch_sha(owner: str, repo: str, branch: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("object", {}).get("sha", "")
    except Exception:
        pass
    return ""

def _create_branch(owner: str, repo: str, new_branch: str, sha: str) -> bool:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
    payload = {"ref": f"refs/heads/{new_branch}", "sha": sha}
    try:
        resp = requests.post(url, headers=_gh_headers(), data=json.dumps(payload), timeout=15)
        return resp.status_code in (200, 201, 422)  # 422 = branch already exists
    except Exception as exc:
        logger.error("Failed to create branch %s: %s", new_branch, exc)
        return False

def _get_file_content_and_sha(owner: str, repo: str, path: str, branch: str) -> tuple[str, str]:
    """Returns (decoded_content, sha)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        resp = requests.get(url, headers=_gh_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
            return content, data.get("sha", "")
    except Exception:
        pass
    return "", ""

def _commit_file(owner: str, repo: str, path: str, new_content: str, sha: str, branch: str, message: str) -> bool:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
    payload = {"message": message, "content": encoded, "sha": sha, "branch": branch}
    try:
        resp = requests.put(url, headers=_gh_headers(), data=json.dumps(payload), timeout=20)
        return resp.status_code in (200, 201)
    except Exception as exc:
        logger.error("Failed to commit %s: %s", path, exc)
        return False

def _create_pull_request(owner: str, repo: str, head_branch: str, base_branch: str,
                          title: str, body: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    payload = {"title": title, "body": body, "head": head_branch, "base": base_branch}
    try:
        resp = requests.post(url, headers=_gh_headers(), data=json.dumps(payload), timeout=15)
        if resp.status_code in (200, 201):
            return resp.json().get("html_url", "")
        elif resp.status_code == 422:
            # PR already exists
            return f"https://github.com/{owner}/{repo}/pulls"
    except Exception as exc:
        logger.error("Failed to create PR: %s", exc)
    return ""

# ── Fix generation ─────────────────────────────────────────────────────────────

SIMPLE_FIXES = {
    "Hardcoded Password":    (r'(password\s*=\s*)["\'][^"\']+["\']', r'\1os.getenv("APP_PASSWORD", "")'),
    "Hardcoded Secret Key":  (r'(secret[_\s]?key\s*=\s*)["\'][^"\']+["\']', r'\1os.getenv("SECRET_KEY", "")'),
    "Hardcoded API Key":     (r'(api[_\s]?key\s*=\s*)["\'][^"\']+["\']', r'\1os.getenv("API_KEY", "")'),
    "Hardcoded Token":       (r'(token\s*=\s*)["\'][A-Za-z0-9_\-]{16,}["\']', r'\1os.getenv("TOKEN", "")'),
    "Unsafe YAML load":      (r'yaml\.load\s*\(([^,)]+)\)', r'yaml.safe_load(\1)'),
    "Debug Mode Enabled":    (r'DEBUG\s*=\s*True', 'DEBUG = False'),
    "MD5 Hash Usage":        (r'hashlib\.md5\s*\(', 'hashlib.sha256('),
    "HTTP (not HTTPS)":      (r'http://', 'https://'),
}

def _apply_simple_fix(content: str, vuln_name: str) -> tuple[str, bool]:
    """Apply regex-based fix. Returns (new_content, changed)."""
    if vuln_name in SIMPLE_FIXES:
        pattern, replacement = SIMPLE_FIXES[vuln_name]
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        return new_content, new_content != content
    return content, False

def _generate_llm_fix(content: str, vuln: dict) -> str:
    """Ask LLM to generate fixed content for complex vulnerabilities."""
    if not openai_client:
        return content
    snippet = vuln.get("snippet", "")
    vuln_name = vuln.get("name", "")
    prompt = f"""You are a cybersecurity engineer fixing Python code.

Vulnerability: {vuln_name}
File content snippet:
```python
{snippet}
```

Return ONLY the fixed version of this exact code snippet (same structure, minimal changes).
Do not add explanations. Just the fixed code."""
    try:
        resp = openai_client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.1
        )
        fixed_snippet = resp.choices[0].message.content.strip().strip("```python").strip("```").strip()
        if fixed_snippet and fixed_snippet != snippet and len(fixed_snippet) < 500:
            return content.replace(snippet, fixed_snippet, 1)
    except Exception:
        pass
    return content

def _generate_diff(original: str, fixed: str, file_path: str) -> str:
    """Generate a simple unified diff string for the PR body."""
    orig_lines = original.splitlines()
    fixed_lines = fixed.splitlines()
    diff_lines = []
    for i, (o, f) in enumerate(zip(orig_lines, fixed_lines)):
        if o != f:
            diff_lines.append(f"- {o}")
            diff_lines.append(f"+ {f}")
    return "\n".join(diff_lines[:30])

# ── Main agent functions ───────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9\-]', '-', text.lower())[:40].strip('-')

def process_repo_fixes(owner: str, report: dict) -> str:
    """Process one repo: apply fixes for HIGH severity vulns and open a PR."""
    repo_name = report.get("repo", "")
    vulns = report.get("vulnerabilities", [])
    
    # Only process HIGH/MEDIUM severity vulns
    actionable = [v for v in vulns if v.get("severity") in ("HIGH", "MEDIUM")]
    if not actionable:
        logger.info("AutoFixPRAgent: No actionable vulns in %s", repo_name)
        return ""
    
    if not GITHUB_TOKEN:
        logger.warning("AutoFixPRAgent: GITHUB_TOKEN not set — cannot create PRs")
        return ""
    
    default_branch = _get_default_branch(owner, repo_name)
    base_sha = _get_branch_sha(owner, repo_name, default_branch)
    if not base_sha:
        return ""
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    fix_branch = f"security-fix-orchestrai-{ts}"
    
    if not _create_branch(owner, repo_name, fix_branch, base_sha):
        logger.error("AutoFixPRAgent: Could not create branch in %s", repo_name)
        return ""
    
    diff_summary = []
    files_fixed = set()
    
    # Group vulns by file
    by_file: dict[str, list[dict]] = {}
    for v in actionable:
        fp = v.get("file", "")
        if fp and fp not in files_fixed:
            by_file.setdefault(fp, []).append(v)
    
    for file_path, file_vulns in list(by_file.items())[:5]:  # Cap at 5 files
        original_content, file_sha = _get_file_content_and_sha(owner, repo_name, file_path, fix_branch)
        if not original_content or not file_sha:
            continue
        
        fixed_content = original_content
        for vuln in file_vulns:
            fixed_content, changed = _apply_simple_fix(fixed_content, vuln.get("name", ""))
            if not changed and vuln.get("snippet"):
                fixed_content = _generate_llm_fix(fixed_content, vuln)
        
        if fixed_content != original_content:
            commit_msg = f"fix(security): {', '.join(v['name'] for v in file_vulns[:2])} in {file_path}"
            ok = _commit_file(owner, repo_name, file_path, fixed_content, file_sha, fix_branch, commit_msg)
            if ok:
                diff_summary.append(_generate_diff(original_content, fixed_content, file_path))
                files_fixed.add(file_path)
                logger.info("AutoFixPRAgent: Fixed %s in %s/%s", file_path, owner, repo_name)
            time.sleep(0.5)
    
    if not files_fixed:
        logger.info("AutoFixPRAgent: No files could be automatically fixed in %s", repo_name)
        return ""
    
    # Create PR
    vuln_names = list({v["name"] for v in actionable})[:3]
    pr_title = f"🔒 Security Fix: {', '.join(vuln_names)}"
    
    pr_body = f"""## 🤖 Automated Security Fix — OrchestrAI

This PR was automatically generated by **OrchestrAI AutoFixPRGeneratorAgent**.

### 🔍 Detected Vulnerabilities
{chr(10).join(f"- **[{v['severity']}]** {v['name']} in `{v['file']}` line {v.get('line', '?')}" for v in actionable[:8])}

### 🔧 Fixes Applied
Files modified: `{"`, `".join(files_fixed)}`

### 📋 Diff Summary
```diff
{chr(10).join(diff_summary[:10])}
```

### ✅ Recommended Next Steps
1. Review the changes in this PR
2. Run your test suite to confirm nothing is broken
3. Merge to remove the security vulnerability
4. Add pre-commit hooks to prevent future hardcoded secrets

---
*Generated by [OrchestrAI](https://orchestrai-agent.onrender.com) on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*
"""
    
    pr_url = _create_pull_request(owner, repo_name, fix_branch, default_branch, pr_title, pr_body)
    if pr_url:
        logger.info("AutoFixPRAgent: ✅ PR created: %s", pr_url)
    return pr_url

def run_auto_fix_pr_agent() -> list[dict]:
    logger.info("AutoFixPRGeneratorAgent: Starting...")
    
    if not GITHUB_TOKEN:
        logger.warning("AutoFixPRGeneratorAgent: GITHUB_TOKEN not set — skipping PR creation")
        return []
    
    # Read security reports
    try:
        data = read_yaml_from_github(SECURITY_REPORTS_FILE) or {}
    except Exception:
        data = {}
    
    reports = data.get("security_reports", [])
    if not reports:
        logger.warning("AutoFixPRGeneratorAgent: No security reports found. Run RepoSecurityScannerAgent first.")
        return []
    
    # Determine owner
    try:
        users_data = read_yaml_from_github("database/users.yaml") or {}
        owner = users_data.get("user", {}).get("github_username", "Swathy1209")
    except Exception:
        owner = "Swathy1209"
    
    results = []
    
    for report in reports:
        risk_level = report.get("risk_level", "Safe")
        if risk_level in ("Safe", "Low"):
            continue  # Skip safe repos
        
        repo_name = report.get("repo", "")
        logger.info("AutoFixPRAgent: Processing %s (Risk: %s)", repo_name, risk_level)
        
        try:
            pr_url = process_repo_fixes(owner, report)
            if pr_url:
                report["auto_fix_pr"] = pr_url
                results.append({"repo": repo_name, "pr_url": pr_url})
                time.sleep(1)
        except Exception as exc:
            logger.error("AutoFixPRAgent: Failed for %s - %s", repo_name, exc)
    
    # Update security reports with PR URLs
    try:
        write_yaml_to_github(SECURITY_REPORTS_FILE, {"security_reports": reports})
    except Exception as exc:
        logger.error("AutoFixPRAgent: Failed to update security reports - %s", exc)
    
    append_log_entry({
        "agent": "AutoFixPRGeneratorAgent",
        "action": f"Generated {len(results)} security fix PRs",
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    })
    logger.info("AutoFixPRGeneratorAgent: Done. %d PRs created.", len(results))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    prs = run_auto_fix_pr_agent()
    print(f"\nCreated {len(prs)} PRs:")
    for p in prs:
        print(f"  {p['repo']}: {p['pr_url']}")
