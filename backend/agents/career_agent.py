"""
career_agent.py — OrchestrAI Career Agent
==========================================
Autonomous agent that:
  1. Fetches AI / Data internships from multiple sources
  2. Filters relevant roles via OpenAI
  3. Stores results in GitHub YAML cloud database
  4. Sends a structured HTML email report daily at 9:30 AM IST
  5. Logs every step to GitHub agent_logs.yaml + execution_history.yaml

GitHub YAML database structure:
  orchestrai-db/
  └── database/
      ├── jobs.yaml
      ├── agent_logs.yaml
      └── execution_history.yaml

Sources:
  - RemoteOK public API
  - Greenhouse/Stripe public API
  - Internshala (scraper)
  - Unstop (scraper)
  - LinkedIn (public job search scraper — no auth required endpoint)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.github_yaml_db import (  # noqa: E402
    append_new_jobs,
    append_log_entry,
    append_execution_record,
    read_jobs_from_github,
)
from backend.email_service import send_email  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CareerAgent")

OPENAI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
openai_client: Optional[OpenAI] = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=0,  # We do our own retry logic via circuit breaker
) if OPENAI_API_KEY else None

# Import shared circuit breaker from ai_engine
from backend.utils.ai_engine import safe_llm_call, _is_daily_quota_error, _mark_quota_exceeded, is_all_quota_exhausted

# ──────────────────────────────────────────────────────────────────────────────
# Domain configuration
# ──────────────────────────────────────────────────────────────────────────────

RELEVANT_DOMAINS: list[str] = [
    "artificial intelligence",
    "machine learning",
    "data science",
    "data analyst",
    "data analysis",
    "business analyst",
    "deep learning",
    "nlp",
    "computer vision",
    "ml engineer",
    "ai engineer",
    "data engineer",
    "research scientist",
]

DOMAIN_KEYWORDS: list[str] = [kw.lower() for kw in RELEVANT_DOMAINS]

TARGET_ROLES: list[str] = [
    "AI Intern",
    "Machine Learning Intern",
    "Data Science Intern",
    "Data Analyst Intern",
    "Business Analyst Intern",
    "Deep Learning Intern",
    "NLP Intern",
    "Computer Vision Intern",
    "Research Intern",
    "ML Engineer Intern",
]

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 20  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_job(
    company: str,
    role: str,
    location: str,
    apply_link: str,
    role_keywords: list[str],
    technical_skills: list[str],
    source: str,
    description: str = "",
) -> dict:
    """Return a canonical internship record with guaranteed company name."""
    safe_company = guarantee_company_name(
        company=company, role=role, apply_link=apply_link, description=description
    )
    return {
        "company": safe_company,
        "role": role.strip(),
        "location": location.strip(),
        "apply_link": apply_link.strip(),
        "role_keywords": [k.strip() for k in role_keywords if k.strip()],
        "technical_skills": [s.strip() for s in technical_skills if s.strip()],
        "source": source,
        "timestamp": _now_iso(),
    }


def _keyword_prefilter(title: str) -> bool:
    """
    Quick keyword-based pre-filter before calling the AI API.
    Returns True if the title contains at least one relevant domain keyword.
    Saves OpenAI credits for obvious non-matches.
    """
    title_lower = title.lower()
    return any(kw in title_lower for kw in DOMAIN_KEYWORDS)


# ── Company Name Extraction (3-layer guarantee) ────────────────────────────────

_KNOWN_JOB_HOSTS = {
    "greenhouse.io", "lever.co", "workday.com", "workdayjobs.com",
    "linkedin.com", "indeed.com", "glassdoor.com", "monster.com",
    "internshala.com", "unstop.com", "naukri.com", "wellfound.com",
    "smartrecruiters.com", "bamboohr.com", "icims.com", "jobvite.com",
    "remoteok.com", "remoteok.io",
}

def _company_from_domain(apply_link: str) -> str:
    """
    Extract company name from the apply link domain.
    Example: https://jobs.tinder.com/apply → 'Tinder'
    """
    if not apply_link:
        return ""
    try:
        parsed = urlparse(apply_link)
        host = parsed.hostname or ""
        # Strip common job-board hosts — they don't indicate the company
        if any(jh in host for jh in _KNOWN_JOB_HOSTS):
            return ""
        # Remove www., jobs., careers., apply. prefixes
        parts = re.sub(r'^(www|jobs|careers|apply|join|work|hiring)\.', '', host).split('.')
        # Take the main domain name (e.g. 'tinder' from 'tinder.com')
        company_raw = parts[0] if parts else ""
        if company_raw and len(company_raw) > 1:
            return company_raw.replace('-', ' ').replace('_', ' ').title()
    except Exception:
        pass
    return ""


def _company_via_llm(role: str, apply_link: str, description: str = "") -> str:
    """Use LLM to extract company name from job metadata."""
    if not openai_client:
        return ""
    prompt = (
        f"Extract only the company name from this job listing. "
        f"Reply with just the company name, nothing else.\n"
        f"Role: {role}\nURL: {apply_link}\nDescription snippet: {description[:200]}"
    )
    try:
        name = safe_llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, temperature=0,
            context="company extraction"
        )
        if name and 2 < len(name) < 60 and name.lower() not in ("unknown", "n/a", "none", ""):
            return name
    except Exception:
        pass
    return ""


def guarantee_company_name(company: str, role: str = "", apply_link: str = "", description: str = "") -> str:
    """
    3-layer company name guarantee:
      1. Use provided name if valid
      2. Extract from apply_link domain
      3. LLM fallback
      4. Mark as 'Unknown Company – Source Incomplete'
    """
    if company and company.strip() and company.strip().lower() not in ("unknown", "", "n/a"):
        return company.strip()
    # Layer 1: domain extraction
    from_domain = _company_from_domain(apply_link)
    if from_domain:
        logger.info("CareerAgent: Extracted company '%s' from domain of %s", from_domain, apply_link)
        return from_domain
    # Layer 2: LLM fallback
    from_llm = _company_via_llm(role, apply_link, description)
    if from_llm:
        logger.info("CareerAgent: LLM extracted company '%s' for role '%s'", from_llm, role)
        return from_llm
    # Layer 3: hard fallback
    return "Unknown Company – Source Incomplete"


def _extract_skills_from_description(description: str) -> tuple[list[str], list[str]]:
    """
    Heuristically extract role keywords and technical skills from a job description.
    Returns (role_keywords, technical_skills).
    """
    skill_patterns = [
        r"\bPython\b", r"\bR\b", r"\bSQL\b", r"\bJava\b", r"\bScala\b",
        r"\bTensorFlow\b", r"\bPyTorch\b", r"\bKeras\b", r"\bScikit-learn\b",
        r"\bscikit.learn\b", r"\bXGBoost\b", r"\bLightGBM\b",
        r"\bSpark\b", r"\bHadoop\b", r"\bAirflow\b", r"\bKafka\b",
        r"\bTableau\b", r"\bPower BI\b", r"\bLooker\b",
        r"\bAWS\b", r"\bAzure\b", r"\bGCP\b", r"\bDocker\b", r"\bKubernetes\b",
        r"\bCUDA\b", r"\bOpenCV\b", r"\bHugging Face\b", r"\bLangChain\b",
        r"\bMLflow\b", r"\bDVC\b", r"\bFastAPI\b", r"\bFlask\b", r"\bDjango\b",
        r"\bPandas\b", r"\bNumPy\b", r"\bMatplotlib\b", r"\bSeaborn\b",
        r"\bJupyter\b", r"\bGit\b", r"\bLinux\b", r"\bBash\b",
    ]
    keyword_patterns = [
        r"\bDeep Learning\b", r"\bMachine Learning\b", r"\bNLP\b",
        r"\bComputer Vision\b", r"\bReinforcement Learning\b",
        r"\bData Science\b", r"\bData Analysis\b", r"\bStatistics\b",
        r"\bPredictive Modeling\b", r"\bNeural Networks?\b",
        r"\bGenerative AI\b", r"\bLLM\b", r"\bModel Deployment\b",
        r"\bFeature Engineering\b", r"\bA/B Testing\b",
        r"\bBusiness Analysis\b", r"\bData Visualization\b",
        r"\bTime Series\b", r"\bAnomaly Detection\b",
    ]

    desc = description or ""
    skills = sorted({
        m.group() for p in skill_patterns
        for m in re.finditer(p, desc, re.IGNORECASE)
    })
    keywords = sorted({
        m.group() for p in keyword_patterns
        for m in re.finditer(p, desc, re.IGNORECASE)
    })
    return keywords[:8], skills[:8]


# ──────────────────────────────────────────────────────────────────────────────
# Source fetchers
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_remoteok_jobs() -> list[dict]:
    """
    Fetch internships from RemoteOK public JSON API.
    Endpoint: https://remoteok.com/api?tag=intern
    """
    logger.info("CareerAgent: Fetching RemoteOK jobs…")
    jobs: list[dict] = []
    url = "https://remoteok.com/api?tag=intern"

    try:
        async with httpx.AsyncClient(headers={**HTTP_HEADERS, "Accept": "application/json"}, timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.json()

        # RemoteOK prepends a notice entry — skip index 0
        for item in raw[1:]:
            title = item.get("position", "")
            if not _keyword_prefilter(title):
                continue

            tags = item.get("tags", [])
            description = item.get("description", "")
            kws, skills = _extract_skills_from_description(description)

            jobs.append(_build_job(
                company=item.get("company", "Unknown"),
                role=title,
                location=item.get("location", "Remote"),
                apply_link=item.get("url", item.get("apply_url", "")),
                role_keywords=kws or tags[:6],
                technical_skills=skills or tags[6:12],
                source="RemoteOK",
            ))

        logger.info("CareerAgent: RemoteOK returned %d pre-filtered jobs.", len(jobs))
    except Exception as exc:
        logger.error("CareerAgent: RemoteOK fetch failed — %s", exc)

    return jobs


async def fetch_stripe_jobs() -> list[dict]:
    """
    Fetch internships from Stripe's Greenhouse public jobs board API.
    Endpoint: https://api.greenhouse.io/v1/boards/stripe/jobs?content=true
    """
    logger.info("CareerAgent: Fetching Stripe Greenhouse jobs…")
    jobs: list[dict] = []
    url = "https://api.greenhouse.io/v1/boards/stripe/jobs?content=true"

    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("jobs", []):
            title = item.get("title", "")
            if "intern" not in title.lower() or not _keyword_prefilter(title):
                continue

            dept = (item.get("departments") or [{}])[0].get("name", "")
            loc_list = item.get("offices") or [{}]
            location = ", ".join(o.get("name", "") for o in loc_list) or "Remote"
            content = item.get("content", "")
            kws, skills = _extract_skills_from_description(content)

            jobs.append(_build_job(
                company="Stripe",
                role=title,
                location=location,
                apply_link=item.get("absolute_url", ""),
                role_keywords=kws or [dept],
                technical_skills=skills,
                source="Stripe Greenhouse",
            ))

        logger.info("CareerAgent: Stripe Greenhouse returned %d pre-filtered jobs.", len(jobs))
    except Exception as exc:
        logger.error("CareerAgent: Stripe Greenhouse fetch failed — %s", exc)

    return jobs


async def _scrape_greenhouse_board(company_slug: str, company_name: str) -> list[dict]:
    """
    Generic Greenhouse board scraper. Reusable for any company that uses Greenhouse.
    """
    jobs: list[dict] = []
    url = f"https://api.greenhouse.io/v1/boards/{company_slug}/jobs?content=true"
    try:
        async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("jobs", []):
            title = item.get("title", "")
            if not _keyword_prefilter(title):
                continue
            loc_list = item.get("offices") or [{}]
            location = ", ".join(o.get("name", "") for o in loc_list) or "Remote"
            content = item.get("content", "")
            kws, skills = _extract_skills_from_description(content)
            jobs.append(_build_job(
                company=company_name,
                role=title,
                location=location,
                apply_link=item.get("absolute_url", ""),
                role_keywords=kws,
                technical_skills=skills,
                source="Greenhouse",
            ))
    except Exception as exc:
        logger.warning("CareerAgent: Greenhouse/%s scrape failed — %s", company_slug, exc)
    return jobs


async def fetch_linkedin_jobs() -> list[dict]:
    """
    Fetch internships from LinkedIn's public jobs search endpoint.
    Uses the unauthenticated public API (no LinkedIn account required).
    Rate-limited; returns best-effort results.
    """
    logger.info("CareerAgent: Fetching LinkedIn jobs…")
    jobs: list[dict] = []

    search_queries = [
        "AI intern",
        "Machine Learning intern",
        "Data Science intern",
        "Data Analyst intern",
        "Business Analyst intern",
    ]

    # LinkedIn public job search URL (no API key needed for basic results)
    base_url = (
        "https://www.linkedin.com/jobs/search/?keywords={query}"
        "&f_JT=I&f_TP=1%2C2%2C3%2C4&sortBy=DD"  # type=Internship, recently posted
    )

    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        for query in search_queries:
            try:
                url = base_url.format(query=quote_plus(query))
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # LinkedIn public results: job cards in <li> with class "jobs-search__results-list"
                cards = soup.select("li.jobs-search__results-list > div") or \
                        soup.select("div.base-card") or \
                        soup.select("li.result-card")

                for card in cards[:10]:  # limit per query
                    title_el = card.select_one("h3, .base-search-card__title, .result-card__title")
                    company_el = card.select_one("h4, .base-search-card__subtitle, .result-card__subtitle")
                    location_el = card.select_one(".job-search-card__location, .base-search-card__metadata")
                    link_el = card.select_one("a[href*='/jobs/view/']")

                    title    = title_el.get_text(strip=True) if title_el else ""
                    company  = company_el.get_text(strip=True) if company_el else "Unknown"
                    location = location_el.get_text(strip=True) if location_el else "Unknown"
                    link     = link_el["href"] if link_el and link_el.has_attr("href") else ""

                    if not title or not _keyword_prefilter(title):
                        continue

                    kws, skills = _extract_skills_from_description(title + " " + query)
                    jobs.append(_build_job(
                        company=company,
                        role=title,
                        location=location,
                        apply_link=link,
                        role_keywords=kws or [query],
                        technical_skills=skills,
                        source="LinkedIn",
                    ))

                # Polite crawl delay
                await asyncio.sleep(1.5)

            except Exception as exc:
                logger.warning("CareerAgent: LinkedIn query '%s' failed — %s", query, exc)

    logger.info("CareerAgent: LinkedIn returned %d pre-filtered jobs.", len(jobs))
    return jobs


async def fetch_internshala_jobs() -> list[dict]:
    """
    Scrape AI/Data internships from Internshala's public listings.
    """
    logger.info("CareerAgent: Fetching Internshala jobs…")
    jobs: list[dict] = []

    categories = [
        ("data-science", "Data Science"),
        ("machine-learning", "Machine Learning"),
        ("artificial-intelligence", "Artificial Intelligence"),
        ("data-analytics", "Data Analytics"),
    ]

    async with httpx.AsyncClient(headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        for slug, label in categories:
            url = f"https://internshala.com/internships/{slug}-internship/"
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")

                # Internshala uses div.individual_internship containers
                cards = soup.select("div.individual_internship") or \
                        soup.select("div[id^='internshiplist']")

                for card in cards[:15]:
                    title_el    = card.select_one("h3.job-internship-name, h3, .heading_4_5")
                    company_el  = card.select_one(".company-name, p.heading_6")
                    location_el = card.select_one(".location_link, span.location_link")
                    link_el     = card.select_one("a.view_detail_button, a[href*='/internship/detail/']")

                    title    = title_el.get_text(strip=True) if title_el else label + " Intern"
                    company  = company_el.get_text(strip=True) if company_el else "Unknown"
                    location = location_el.get_text(strip=True) if location_el else "India / Remote"
                    link_href = link_el["href"] if link_el and link_el.has_attr("href") else ""
                    link = f"https://internshala.com{link_href}" if link_href.startswith("/") else link_href

                    kws, skills = _extract_skills_from_description(title)
                    jobs.append(_build_job(
                        company=company,
                        role=title,
                        location=location,
                        apply_link=link,
                        role_keywords=kws or [label],
                        technical_skills=skills,
                        source="Internshala",
                    ))

                await asyncio.sleep(1)

            except Exception as exc:
                logger.warning("CareerAgent: Internshala category '%s' failed — %s", slug, exc)

    logger.info("CareerAgent: Internshala returned %d pre-filtered jobs.", len(jobs))
    return jobs


async def fetch_unstop_jobs() -> list[dict]:
    """
    Fetch internships from Unstop public API / scraper.
    Unstop exposes a public opportunities listing.
    """
    logger.info("CareerAgent: Fetching Unstop jobs…")
    jobs: list[dict] = []

    # Unstop public opportunities endpoint (returns JSON)
    domains = ["data-science", "machine-learning", "artificial-intelligence"]
    base_api = "https://unstop.com/api/public/opportunity/search-result?opportunity=jobs&per_page=20"

    async with httpx.AsyncClient(headers={**HTTP_HEADERS, "Accept": "application/json"}, timeout=REQUEST_TIMEOUT) as client:
        for domain in domains:
            try:
                url = f"{base_api}&searchText={quote_plus(domain)}&oppType=internship"
                resp = await client.get(url)
                if resp.status_code != 200:
                    # Fall back to HTML scrape
                    html_url = f"https://unstop.com/internships?domain={domain}"
                    html_resp = await client.get(html_url)
                    if html_resp.status_code != 200:
                        continue

                    soup = BeautifulSoup(html_resp.text, "html.parser")
                    cards = soup.select("div.opportunity-card, div.card-item, article")
                    for card in cards[:10]:
                        title_el   = card.select_one("h2, h3, .title")
                        company_el = card.select_one(".company-name, .org-name")
                        link_el    = card.select_one("a[href]")
                        title   = title_el.get_text(strip=True) if title_el else ""
                        company = company_el.get_text(strip=True) if company_el else "Unknown"
                        href    = link_el["href"] if link_el else ""
                        link    = f"https://unstop.com{href}" if href.startswith("/") else href
                        if title and _keyword_prefilter(title):
                            kws, skills = _extract_skills_from_description(title)
                            jobs.append(_build_job(
                                company=company,
                                role=title,
                                location="India / Remote",
                                apply_link=link,
                                role_keywords=kws or [domain.replace("-", " ").title()],
                                technical_skills=skills,
                                source="Unstop",
                            ))
                    continue

                data = resp.json().get("data", {})
                items = data.get("data", []) if isinstance(data, dict) else []
                for item in items:
                    title   = item.get("title", "")
                    company = (item.get("organisation") or {}).get("name", "Unknown")
                    location = item.get("city", "India")
                    link    = f"https://unstop.com/jobs/{item.get('id', '')}"
                    if not _keyword_prefilter(title):
                        continue
                    kws, skills = _extract_skills_from_description(
                        title + " " + item.get("short_description", "")
                    )
                    jobs.append(_build_job(
                        company=company,
                        role=title,
                        location=location,
                        apply_link=link,
                        role_keywords=kws,
                        technical_skills=skills,
                        source="Unstop",
                    ))

                await asyncio.sleep(1)

            except Exception as exc:
                logger.warning("CareerAgent: Unstop domain '%s' failed — %s", domain, exc)

    logger.info("CareerAgent: Unstop returned %d pre-filtered jobs.", len(jobs))
    return jobs


async def _fetch_all_sources() -> list[dict]:
    """
    Run all source fetchers concurrently and merge results.
    """
    results = await asyncio.gather(
        fetch_remoteok_jobs(),
        fetch_linkedin_jobs(),
        fetch_stripe_jobs(),
        fetch_internshala_jobs(),
        fetch_unstop_jobs(),
        return_exceptions=False,
    )
    all_jobs: list[dict] = []
    for batch in results:
        if isinstance(batch, list):
            all_jobs.extend(batch)
    return all_jobs


# ──────────────────────────────────────────────────────────────────────────────
# AI Relevance Filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_jobs_ai(jobs: list[dict]) -> list[dict]:
    """
    Filter jobs using a SINGLE batched Gemini API call instead of one call per job.
    This eliminates 429 rate limit errors by going from 100+ calls → 2-3 calls max.
    Falls back to keyword filter if API is unavailable or quota exceeded.
    """
    logger.info("CareerAgent: Filtering %d jobs via AI relevance check (batched)…", len(jobs))

    # Import here to get the live module-level flag value each time
    import backend.utils.ai_engine as _ai_eng

    if not openai_client or not jobs or _ai_eng.is_all_quota_exhausted():
        logger.info("CareerAgent: No LLM / quota exceeded — using keyword fallback for all jobs.")
        return [j for j in jobs if _keyword_prefilter(j.get("role", ""))]

    BATCH = 50
    relevant: list[dict] = []
    MODELS = ["gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]

    for start in range(0, len(jobs), BATCH):
        # Check circuit breaker before each batch
        if _ai_eng.is_all_quota_exhausted():
            logger.warning("CareerAgent: Circuit breaker open — keyword fallback for remaining batches.")
            relevant.extend([j for j in jobs[start:] if _keyword_prefilter(j.get("role", ""))])
            break

        batch = jobs[start: start + BATCH]
        lines = "\n".join(
            f"{i+1}. {j.get('role','')} @ {j.get('company','')} | skills: {', '.join(j.get('technical_skills',[])[:3])}"
            for i, j in enumerate(batch)
        )
        prompt = (
            "You are a career relevance classifier.\n"
            "For each internship below, reply with its NUMBER if it is relevant to:\n"
            "AI, Machine Learning, Data Science, Data Engineering, Data Analyst, "
            "Business Analyst, NLP, Computer Vision, or Deep Learning.\n\n"
            f"{lines}\n\n"
            "Reply ONLY with comma-separated numbers of relevant internships. "
            "Example: 1,3,5,7\n"
            "If none are relevant, reply: NONE"
        )

        answer = safe_llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0,
            context=f"job filter batch {start+1}-{start+len(batch)}",
        )

        if answer is None:
            # Circuit breaker or all models failed — keyword fallback for this batch
            logger.warning("CareerAgent: LLM unavailable — keyword fallback for batch %d-%d.", start+1, start+len(batch))
            relevant.extend([j for j in batch if _keyword_prefilter(j.get("role", ""))])
        elif answer.upper() == "NONE":
            pass  # No relevant jobs in this batch
        else:
            logger.info("CareerAgent: Batch %d-%d LLM answer: %s", start+1, start+len(batch), answer[:80])
            indices = set()
            for token in answer.replace(" ", "").split(","):
                try:
                    idx = int(token)
                    if 1 <= idx <= len(batch):
                        indices.add(idx - 1)
                except ValueError:
                    pass
            for idx in sorted(indices):
                relevant.append(batch[idx])

        if start + BATCH < len(jobs) and not _ai_eng.is_all_quota_exhausted():
            time.sleep(1)  # Reduced from 2s to 1s — breaker handles quota errors now

    logger.info("CareerAgent: %d / %d jobs passed relevance filter.", len(relevant), len(jobs))
    return relevant


# Backward-compat alias
filter_relevant_jobs_ai = filter_jobs_ai



# ──────────────────────────────────────────────────────────────────────────────
# GitHub YAML storage
# ──────────────────────────────────────────────────────────────────────────────

def store_jobs_yaml_github(jobs: list[dict]) -> tuple[int, int]:
    """
    Append new relevant jobs to GitHub YAML cloud database (database/jobs.yaml).
    Returns (added, total).
    """
    logger.info("CareerAgent: Storing %d relevant jobs to GitHub YAML…", len(jobs))
    added, total = append_new_jobs(jobs)
    logger.info("CareerAgent: Stored in GitHub YAML — %d new, %d total.", added, total)
    return added, total


# Backward-compat alias
store_jobs_github = store_jobs_yaml_github


# ──────────────────────────────────────────────────────────────────────────────
# Email helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_jobs_email(jobs: list[dict]) -> str:
    """
    Return a plain-text representation of jobs for logging/debugging.
    The full HTML/plain-text email is built in email_service.py.
    """
    lines = []
    for job in jobs:
        lines.append(
            f"Company: {job['company']}\n"
            f"Role: {job['role']}\n"
            f"Location: {job['location']}\n"
            f"Apply Link: {job['apply_link']}\n\n"
            f"Role Keywords:\n{', '.join(job.get('role_keywords', []))}\n\n"
            f"Technical Skills Required:\n{', '.join(job.get('technical_skills', []))}\n"
            + "-" * 60
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Logging to GitHub YAML
# ──────────────────────────────────────────────────────────────────────────────

def _log_to_github(action: str, level: str = "INFO") -> None:
    """
    Append a structured log entry to GitHub database/agent_logs.yaml.

    YAML entry format:
      - agent:     CareerAgent
        action:    Fetch started
        timestamp: 2026-03-01T09:30:00
    """
    try:
        append_log_entry({
            "agent":     "CareerAgent",
            "action":    action,
            "level":     level,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception as exc:
        # Never let logging failures crash the agent
        logger.warning("CareerAgent: GitHub YAML log write failed — %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_career_agent() -> dict:
    """
    Main orchestration function.
    Full pipeline:
      fetch → filter (AI) → convert to YAML → store in GitHub jobs.yaml
      → append agent_logs.yaml → send email → record execution_history.yaml

    Returns a summary dict with counts and status.
    """
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    summary: dict = {
        "status":      "started",
        "run_at":      run_at,
        "fetched":     0,
        "relevant":    0,
        "stored_new":  0,
        "total_in_db": 0,
        "email_sent":  False,
        "errors":      [],
    }

    logger.info("CareerAgent: ── Pipeline started ──")
    _log_to_github("Fetch started")

    # ── Step 1: Fetch (async, all sources concurrently) ───────────────────────
    all_jobs: list[dict] = []
    try:
        all_jobs = asyncio.run(_fetch_all_sources())
        summary["fetched"] = len(all_jobs)
        action = f"{len(all_jobs)} jobs fetched"
        logger.info("CareerAgent: %s", action)
        _log_to_github(action)
    except Exception as exc:
        err = f"Fetch step failed: {exc}"
        logger.error("CareerAgent: %s", err)
        summary["errors"].append(err)
        _log_to_github(f"FETCH ERROR — {exc}", level="ERROR")
        summary["status"] = "partial_failure"

    # ── Step 2: Filter via AI (OpenAI GPT / keyword fallback) ────────────────
    relevant_jobs: list[dict] = []
    try:
        relevant_jobs = filter_jobs_ai(all_jobs)
        summary["relevant"] = len(relevant_jobs)
        action = f"{len(all_jobs)} jobs fetched, {len(relevant_jobs)} relevant"
        logger.info("CareerAgent: %s", action)
        _log_to_github(action)
    except Exception as exc:
        err = f"Filter step failed: {exc}"
        logger.error("CareerAgent: %s", err)
        summary["errors"].append(err)
        _log_to_github(f"FILTER ERROR — {exc}", level="ERROR")
        relevant_jobs = all_jobs  # degrade gracefully — store all if filter fails

    # ── Step 3: Convert to YAML + store in GitHub jobs.yaml ──────────────────
    if relevant_jobs:
        try:
            added, total = store_jobs_yaml_github(relevant_jobs)
            summary["stored_new"]  = added
            summary["total_in_db"] = total
            _log_to_github("Stored in GitHub YAML")
        except Exception as exc:
            err = f"Store step failed: {exc}"
            logger.error("CareerAgent: %s", err)
            summary["errors"].append(err)
            _log_to_github(f"STORE ERROR — {exc}", level="ERROR")

    # ── Step 4: Email delegated to ExecutionAgent ─────────────────────────────
    # Reads all stored jobs so the email covers the full catalogue
    summary["email_sent"] = False  # Legacy field, handled downstream now
    logger.info("CareerAgent: Email delegation passed to ExecutionAgent")
    _log_to_github("Email delegation passed to ExecutionAgent")

    # ── Final status ──────────────────────────────────────────────────────────
    if not summary["errors"]:
        summary["status"] = "success"
    elif summary["email_sent"]:
        summary["status"] = "partial_success"

    # ── Step 5: Record execution history in GitHub ────────────────────────────
    try:
        append_execution_record({
            "run_at":      run_at,
            "fetched":     summary["fetched"],
            "relevant":    summary["relevant"],
            "stored_new":  summary["stored_new"],
            "total_in_db": summary["total_in_db"],
            "email_sent":  summary["email_sent"],
            "status":      summary["status"],
            "errors":      summary["errors"],
        })
    except Exception as exc:
        logger.warning("CareerAgent: Could not write execution history — %s", exc)

    logger.info("CareerAgent: ── Pipeline complete: %s ──", summary["status"].upper())
    logger.info(
        "CareerAgent: Summary\n%s",
        yaml.dump(summary, default_flow_style=False, sort_keys=False),
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler hook
# ──────────────────────────────────────────────────────────────────────────────

def schedule_daily_execution() -> None:
    """
    Entry point to start the APScheduler-based daily scheduler.
    Delegates to backend.scheduler.schedule_daily_internship_email.
    """
    from backend.scheduler import schedule_daily_internship_email  # local import to avoid circular
    schedule_daily_internship_email(run_career_agent, hour=9, minute=30)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OrchestrAI Career Agent")
    parser.add_argument(
        "--mode",
        choices=["run", "schedule"],
        default="run",
        help=(
            "run     — Execute the pipeline once immediately (default)\n"
            "schedule — Start the daily APScheduler loop (9:30 AM IST)"
        ),
    )
    args = parser.parse_args()

    if args.mode == "schedule":
        schedule_daily_execution()
    else:
        result = run_career_agent()
        sys.exit(0 if result["status"] in ("success", "partial_success") else 1)
