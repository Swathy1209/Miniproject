"""
resume_parser.py — Resume PDF Download & Text Extraction
OrchestrAI Autonomous Multi-Agent System

Responsibilities:
  - Download resume PDF from GitHub repository via REST API
  - Save temporarily to disk
  - Extract full text using PyPDF2
  - Clean up temp file after extraction
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("OrchestrAI.ResumeParser")

GITHUB_TOKEN:    str = os.getenv("GITHUB_TOKEN", "")
GITHUB_USERNAME: str = os.getenv("GITHUB_USERNAME", "")
GITHUB_REPO:     str = os.getenv("GITHUB_REPO", "orchestrai-db")
GITHUB_BRANCH:   str = os.getenv("GITHUB_BRANCH", "main")

_REPO_SLUG: str = (
    GITHUB_REPO if "/" in GITHUB_REPO else f"{GITHUB_USERNAME}/{GITHUB_REPO}"
)
_BASE_URL = "https://api.github.com"


def _auth_headers() -> dict:
    if not GITHUB_TOKEN:
        raise EnvironmentError("GITHUB_TOKEN is not set.")
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def download_resume_from_github(
    resume_path: str = "resumes/swathiga_resume.pdf",
    local_path:  str = "temp_resume.pdf",
) -> str | None:
    """
    Download the resume PDF from GitHub repo via REST API.
    Falls back to checking local Disk if GitHub fails or is not configured.

    Args:
        resume_path: Path inside the GitHub repo  (e.g. "resumes/swathiga_resume.pdf")
        local_path:  Where to save it locally     (e.g. "temp_resume.pdf")

    Returns:
        Absolute path to the downloaded/found file, or None on failure.
    """
    
    # ── Try Local Check First (Fallback/Development) ──────────────────────────
    # Check project-relative and current directory paths
    possible_local_paths = [
        resume_path,
        os.path.join("MultiAgent_Project", resume_path),
        os.path.join("backend", resume_path),
        Path(resume_path).name # just the filename
    ]
    
    for p in possible_local_paths:
        if os.path.exists(p):
            logger.info("ResumeParser: Found local resume at '%s' — bypassing GitHub fetch.", p)
            # Copy to temp_resume.pdf if it's not already there
            if str(Path(p).resolve()) != str(Path(local_path).resolve()):
                Path(local_path).write_bytes(Path(p).read_bytes())
            return str(Path(local_path).resolve())

    # ── Try GitHub Fetch ───────────────────────────────────────────────────────
    if GITHUB_TOKEN and GITHUB_USERNAME and GITHUB_REPO:
        url = f"{_BASE_URL}/repos/{_REPO_SLUG}/contents/{resume_path}"
        logger.info("ResumeParser: Attempting to download '%s' from GitHub...", resume_path)

        try:
            resp = requests.get(
                url,
                headers=_auth_headers(),
                params={"ref": GITHUB_BRANCH},
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                # GitHub may return content in chunks for large files (download_url)
                if data.get("encoding") == "base64" and data.get("content"):
                    pdf_bytes = base64.b64decode(data["content"])
                elif data.get("download_url"):
                    # Fallback: direct download for large files
                    dl_resp = requests.get(data["download_url"], timeout=60)
                    dl_resp.raise_for_status()
                    pdf_bytes = dl_resp.content
                else:
                    logger.error("ResumeParser: Could not decode file content from GitHub response.")
                    return None

                abs_path = str(Path(local_path).resolve())
                Path(local_path).write_bytes(pdf_bytes)
                logger.info("ResumeParser: Resume saved to '%s' (%d bytes).", abs_path, len(pdf_bytes))
                return abs_path
            
            elif resp.status_code == 404:
                logger.warning("ResumeParser: Resume not found on GitHub at '%s'.", resume_path)
            else:
                logger.warning("ResumeParser: GitHub responded with %d", resp.status_code)

        except Exception as exc:
            logger.error("ResumeParser: download_resume_from_github() failed — %s", exc)
    
    logger.error("ResumeParser: ❌ Resume NOT FOUND locally or on GitHub.")
    return None


def extract_resume_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyPDF2.

    Args:
        pdf_path: Local path to the PDF file.

    Returns:
        Extracted text as a single string. Empty string on failure.
    """
    try:
        import PyPDF2  # noqa: PLC0415
    except ImportError:
        logger.error("ResumeParser: PyPDF2 not installed. Run: pip install PyPDF2")
        return ""

    try:
        text_parts = []
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            total_pages = len(reader.pages)
            logger.info("ResumeParser: PDF has %d pages.", total_pages)

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                except Exception as page_exc:
                    logger.warning("ResumeParser: Page %d extraction failed — %s", i, page_exc)

        full_text = "\n".join(text_parts).strip()
        logger.info("ResumeParser: Extracted %d characters from resume.", len(full_text))
        return full_text

    except FileNotFoundError:
        logger.error("ResumeParser: PDF not found at '%s'.", pdf_path)
        return ""
    except Exception as exc:
        logger.error("ResumeParser: extract_resume_text() failed — %s", exc)
        return ""


def download_and_extract(
    resume_path: str = "resumes/swathiga_resume.pdf",
    local_path:  str = "temp_resume.pdf",
    cleanup:     bool = True,
) -> str:
    """
    Convenience function: download resume from GitHub and extract text.

    Args:
        resume_path: Path in GitHub repo.
        local_path:  Temp local filename.
        cleanup:     Delete temp PDF after extraction (default True).

    Returns:
        Extracted resume text, or empty string on failure.
    """
    saved_path = download_resume_from_github(resume_path, local_path)
    if not saved_path:
        return ""

    text = extract_resume_text(saved_path)

    if cleanup:
        try:
            Path(saved_path).unlink(missing_ok=True)
            logger.info("ResumeParser: Temp file '%s' deleted.", saved_path)
        except Exception as exc:
            logger.warning("ResumeParser: Could not delete temp file — %s", exc)

    return text
