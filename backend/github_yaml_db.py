"""
github_yaml_db.py — GitHub REST API YAML Cloud Database Handler
OrchestrAI Autonomous Multi-Agent System

Stores all data as human-readable YAML files inside:
  orchestrai-db/
  └── database/
      ├── jobs.yaml
      ├── agent_logs.yaml
      └── execution_history.yaml

Public API
----------
read_yaml_from_github(file_path)       → dict
write_yaml_to_github(file_path, data)  → bool
append_to_yaml(file_path, new_entry)   → bool
update_yaml(file_path, updated_data)   → bool

append_new_jobs(new_jobs)              → (int, int)
read_jobs_from_github()                → list[dict]
append_log_entry(entry)                → bool
append_execution_record(record)        → bool
read_logs_from_github()                → list[dict]
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("CareerAgent.GitHubYAMLDB")

# ── Config ─────────────────────────────────────────────────────────────────────
GITHUB_TOKEN:    str = os.getenv("GITHUB_TOKEN", "")
GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME", "")
GITHUB_REPO:     str = os.getenv("GITHUB_REPO", "orchestrai-db")
GITHUB_BRANCH:   str = os.getenv("GITHUB_BRANCH", "main")

# Fully-qualified repo slug: "username/repo"  (supports both formats)
_REPO_SLUG: str = (
    GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
)

# YAML file paths inside the repo
JOBS_FILE:      str = "database/jobs.yaml"
LOGS_FILE:      str = "database/agent_logs.yaml"
HISTORY_FILE:   str = "database/execution_history.yaml"

_BASE_URL = "https://api.github.com"

# ── YAML dumper that preserves nice formatting ──────────────────────────────────
def _yaml_dumps(data: Any) -> str:
    """
    Serialize `data` to a nicely formatted YAML string.
    - allow_unicode keeps non-ASCII readable
    - default_flow_style=False forces block style (no inline lists)
    - sort_keys=False preserves insertion order
    """
    return yaml.dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
    )


# ── Internal Local Space helpers ───────────────────────────────────────────────

DATA_DIR = os.getenv("DATA_DIR", ".")
try:
    os.makedirs(os.path.join(DATA_DIR, "database"), exist_ok=True)
except PermissionError:
    DATA_DIR = "./data"
    os.makedirs(os.path.join(DATA_DIR, "database"), exist_ok=True)

def _auth_headers() -> dict[str, str]:
    return {}

def _get_raw_file(file_path: str) -> tuple[str, str]:
    full_path = os.path.join(DATA_DIR, file_path)
    if not os.path.exists(full_path):
        logger.info("LocalDB: '%s' does not exist yet — will create.", file_path)
        return "", ""

    with open(full_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return raw, ""

def _put_raw_file(file_path: str, text: str, sha: str, commit_msg: str) -> bool:
    full_path = os.path.join(DATA_DIR, file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("LocalDB: '%s' saved — %s", file_path, commit_msg)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — generic YAML helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_yaml_from_github(file_path: str) -> dict:
    """
    Read and parse a YAML file from the GitHub repo.

    Returns a Python dict/list (whatever the YAML top level is).
    Returns an empty dict {} if the file does not exist yet.
    """
    try:
        raw, _ = _get_raw_file(file_path)
        if not raw:
            return {}
        parsed = yaml.safe_load(raw) or {}
        logger.debug("GitHubYAMLDB: read '%s'.", file_path)
        return parsed
    except Exception as exc:
        logger.error("GitHubYAMLDB: read_yaml_from_github('%s') failed — %s", file_path, exc)
        return {}


def write_yaml_to_github(file_path: str, yaml_data: Any) -> bool:
    """
    Fully overwrite a YAML file in the GitHub repo.

    ⚠️  Prefer append_to_yaml() / update_yaml() for incremental changes.
    """
    try:
        _, sha = _get_raw_file(file_path)
        text = _yaml_dumps(yaml_data)
        ts   = datetime.now(timezone.utc).isoformat()
        return _put_raw_file(
            file_path, text, sha,
            f"chore({file_path}): full overwrite — {ts}",
        )
    except Exception as exc:
        logger.error("GitHubYAMLDB: write_yaml_to_github('%s') failed — %s", file_path, exc)
        return False


def append_to_yaml(file_path: str, new_entry: dict) -> bool:
    """
    Append a single dict entry to a top-level YAML list.

    Expected file structure: a YAML mapping with ONE list key
    whose name is inferred from `file_path`.

    Example:
      file_path = "database/agent_logs.yaml"
      → top-level key = "logs"
      → appends new_entry to data["logs"]
    """
    _KEY_MAP = {
        JOBS_FILE:    "jobs",
        LOGS_FILE:    "logs",
        HISTORY_FILE: "executions",
    }
    list_key = _KEY_MAP.get(file_path, "entries")

    try:
        raw, sha = _get_raw_file(file_path)
        data: dict = yaml.safe_load(raw) if raw else {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault(list_key, [])
        data[list_key].append(new_entry)

        text = _yaml_dumps(data)
        ts   = datetime.now(timezone.utc).isoformat()
        return _put_raw_file(
            file_path, text, sha,
            f"feat({file_path}): append entry — {ts}",
        )
    except Exception as exc:
        logger.error("GitHubYAMLDB: append_to_yaml('%s') failed — %s", file_path, exc)
        return False


def update_yaml(file_path: str, updated_data: Any) -> bool:
    """
    Alias for write_yaml_to_github that signals intentional partial update.
    Reads current SHA internally so callers don't need to manage it.
    """
    return write_yaml_to_github(file_path, updated_data)


# ══════════════════════════════════════════════════════════════════════════════
# Jobs API
# ══════════════════════════════════════════════════════════════════════════════

def read_jobs_from_github() -> list[dict]:
    """
    Read all internship records from database/jobs.yaml.
    Returns a list of job dicts (empty list if file doesn't exist).
    """
    try:
        data = read_yaml_from_github(JOBS_FILE)
        jobs = data.get("jobs", [])
        if not isinstance(jobs, list):
            logger.warning("GitHubYAMLDB: jobs.yaml 'jobs' key is not a list — resetting.")
            return []
        logger.info("GitHubYAMLDB: Read %d existing jobs from YAML.", len(jobs))
        return jobs
    except Exception as exc:
        logger.error("GitHubYAMLDB: read_jobs_from_github failed — %s", exc)
        return []


def append_new_jobs(new_jobs: list[dict]) -> tuple[int, int]:
    """
    Merge new_jobs into jobs.yaml without overwriting existing entries.

    Deduplication key: company + role + source (case-insensitive).
    Returns (jobs_added, total_jobs_in_db).
    """
    try:
        raw, sha = _get_raw_file(JOBS_FILE)
        data: dict = yaml.safe_load(raw) if raw else {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("jobs", [])

        existing: list[dict] = data["jobs"]
        existing_keys: set[str] = {
            f"{j.get('company','').lower()}|{j.get('role','').lower()}|{j.get('source','').lower()}"
            for j in existing
        }

        added = 0
        for job in new_jobs:
            key = (
                f"{job.get('company','').lower()}"
                f"|{job.get('role','').lower()}"
                f"|{job.get('source','').lower()}"
            )
            if key not in existing_keys:
                existing.append(job)
                existing_keys.add(key)
                added += 1

        total = len(existing)

        if added > 0:
            data["jobs"] = existing
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            text = _yaml_dumps(data)
            ts   = datetime.now(timezone.utc).isoformat()
            _put_raw_file(
                JOBS_FILE, text, sha,
                f"feat(jobs): +{added} new internships — {ts}",
            )
            logger.info("GitHubYAMLDB: Appended %d new jobs. Total: %d.", added, total)
        else:
            logger.info("GitHubYAMLDB: No new jobs — total unchanged at %d.", total)

        return added, total

    except Exception as exc:
        logger.error("GitHubYAMLDB: append_new_jobs failed — %s", exc)
        return 0, 0


# ══════════════════════════════════════════════════════════════════════════════
# Logs API
# ══════════════════════════════════════════════════════════════════════════════

def append_log_entry(entry: dict) -> bool:
    """
    Append a structured log entry to database/agent_logs.yaml.

    Entry format (all fields optional except 'action'):
      {
        "agent":     "CareerAgent",
        "action":    "Fetch started",
        "timestamp": "2026-03-01T09:30:00"   ← auto-filled if absent
      }
    Keeps last 500 entries to prevent unbounded growth.
    """
    try:
        raw, sha = _get_raw_file(LOGS_FILE)
        data: dict = yaml.safe_load(raw) if raw else {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("logs", [])

        entry.setdefault("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        data["logs"].append(entry)
        data["logs"] = data["logs"][-500:]   # rolling window

        text = _yaml_dumps(data)
        ts   = datetime.now(timezone.utc).isoformat()
        return _put_raw_file(
            LOGS_FILE, text, sha,
            f"log(agent): {entry.get('action', 'activity')} — {ts}",
        )
    except Exception as exc:
        logger.error("GitHubYAMLDB: append_log_entry failed — %s", exc)
        return False


def read_logs_from_github() -> list[dict]:
    """Read all log entries from database/agent_logs.yaml."""
    try:
        data = read_yaml_from_github(LOGS_FILE)
        return data.get("logs", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.error("GitHubYAMLDB: read_logs_from_github failed — %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Execution History API
# ══════════════════════════════════════════════════════════════════════════════

def append_execution_record(record: dict) -> bool:
    """
    Append a pipeline execution summary to database/execution_history.yaml.

    Typical record:
      {
        "run_at":     "2026-03-01T09:30:00",
        "fetched":    24,
        "relevant":   12,
        "stored_new": 5,
        "total_in_db": 87,
        "email_sent": true,
        "status":     "success",
        "errors":     []
      }
    """
    try:
        raw, sha = _get_raw_file(HISTORY_FILE)
        data: dict = yaml.safe_load(raw) if raw else {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("executions", [])

        record.setdefault("run_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
        data["executions"].append(record)
        data["executions"] = data["executions"][-200:]  # keep last 200 runs

        text = _yaml_dumps(data)
        ts   = datetime.now(timezone.utc).isoformat()
        return _put_raw_file(
            HISTORY_FILE, text, sha,
            f"log(execution): run recorded — {ts}",
        )
    except Exception as exc:
        logger.error("GitHubYAMLDB: append_execution_record failed — %s", exc)
        return False
