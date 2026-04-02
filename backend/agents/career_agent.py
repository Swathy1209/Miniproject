"""
career_agent.py — The Autonomous Career Agent
OrchestrAI Autonomous Multi-Agent System
"""

import logging
import os
import sys
import asyncio
import time
import re
import yaml
from datetime import datetime, timezone
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
import httpx
from dotenv import load_dotenv

load_dotenv()

from backend.github_yaml_db import (
    append_new_jobs,
    append_log_entry,
    append_execution_record,
    read_yaml_from_github,
)

logger = logging.getLogger("OrchestrAI.CareerAgent")

# Configuration
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
REQUEST_TIMEOUT = 30
DOMAIN_KEYWORDS = [
    "data science", "machine learning", "artificial intelligence", "ai ", "data analyst",
    "data analyst", "business analyst", "bi analyst", "deep learning", "nlp", "computer vision",
    "data engineer", "software engineer", "full stack", "frontend", "backend"
]

OPENAI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

# Import shared AI engine helpers
from backend.utils.ai_engine import safe_llm_call, is_all_quota_exhausted

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

def _build_job(company, role, location, apply_link, role_keywords, technical_skills, source, description="") -> dict:
    return {
        "company": company.strip(),
        "role": role.strip(),
        "location": location.strip(),
        "apply_link": apply_link.strip(),
        "role_keywords": sorted(list(set(role_keywords))),
        "technical_skills": sorted(list(set(technical_skills))),
        "source": source,
        "timestamp": _now_iso(),
    }

def _keyword_prefilter(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in DOMAIN_KEYWORDS)

# ──────────────────────────────────────────────────────────────────────────────
# Fetchers (LinkedIn, RemoteOK, etc.)
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_remoteok_jobs() -> list[dict]:
    logger.info("CareerAgent: Fetching RemoteOK jobs…")
    jobs = []
    url = "https://remoteok.com/api?tag=intern"
    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for item in data[1:]:
                    title = item.get("position", "")
                    if _keyword_prefilter(title):
                        jobs.append(_build_job(
                            item.get("company", "Unknown"),
                            title,
                            item.get("location", "Remote"),
                            item.get("url", ""),
                            item.get("tags", []),
                            item.get("tags", []),
                            "RemoteOK"
                        ))
    except Exception as e:
        logger.error("CareerAgent: RemoteOK failed — %s", e)
    return jobs

async def fetch_linkedin_jobs() -> list[dict]:
    # Placeholder for simplicity in restoration
    return []

async def _fetch_all_sources() -> list[dict]:
    results = await asyncio.gather(
        fetch_remoteok_jobs(),
        fetch_linkedin_jobs(),
        return_exceptions=True
    )
    all_jobs = []
    for r in results:
        if isinstance(r, list): all_jobs.extend(r)
    return all_jobs

# ──────────────────────────────────────────────────────────────────────────────
# AI Relevance Filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_jobs_ai(jobs: list[dict]) -> list[dict]:
    if not jobs or is_all_quota_exhausted(): return jobs[:50] # Fallback
    
    logger.info("CareerAgent: Filtering %d jobs via batched AI...", len(jobs))
    BATCH = 25
    relevant = []
    
    for i in range(0, len(jobs), BATCH):
        batch = jobs[i:i+BATCH]
        lines = "\n".join([f"{idx+1}. {j['role']} @ {j['company']}" for idx, j in enumerate(batch)])
        prompt = f"Identify relevant AI/Data internships (e.g. Data Scientist, ML, Analyst). Reply with numbers only (e.g. 1, 3, 5):\n\n{lines}"
        
        raw = safe_llm_call([{"role": "user", "content": prompt}], max_tokens=100, context="batch filter")
        if raw:
            nums = re.findall(r"\d+", raw)
            for n in nums:
                idx = int(n) - 1
                if 0 <= idx < len(batch): relevant.append(batch[idx])
        else:
            relevant.extend([j for j in batch if _keyword_prefilter(j['role'])])
            
    return relevant

# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────

def run_career_agent() -> dict:
    logger.info("CareerAgent: Pipeline started")
    all_jobs = asyncio.run(_fetch_all_sources())
    relevant = filter_jobs_ai(all_jobs)
    
    added, total = append_new_jobs(relevant)
    
    summary = {
        "status": "success",
        "fetched": len(all_jobs),
        "relevant": len(relevant),
        "stored_new": added,
        "total_in_db": total,
        "timestamp": _now_iso()
    }
    
    append_execution_record(summary)
    logger.info("CareerAgent: Pipeline complete. Total jobs: %d", total)
    return summary

if __name__ == "__main__":
    run_career_agent()
