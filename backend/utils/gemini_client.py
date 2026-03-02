"""
gemini_client.py — Google Gemini AI Client (OpenAI-compatible)
OrchestrAI Autonomous Multi-Agent System

Uses Gemini's OpenAI-compatible REST endpoint so all existing OpenAI
SDK calls work without changes — just swap the client.

Free tier: 15 requests/min, 1,500 requests/day, 1M tokens/day
Model: gemini-1.5-flash (fast, free, powerful)
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("OrchestrAI.GeminiClient")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Gemini's OpenAI-compatible base URL
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Default model — fast, free, capable
GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_client() -> OpenAI | None:
    """
    Return an OpenAI client pointed at Gemini's compatible endpoint.
    Returns None if no API key is configured.
    """
    if not GEMINI_API_KEY:
        logger.warning("GeminiClient: No API key found — AI features will use fallbacks.")
        return None
    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
    )


def ai_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.7,
    client: OpenAI | None = None,
) -> str:
    """
    Send a chat completion request via Gemini (OpenAI-compatible).
    Returns the response text, or empty string on failure.

    Args:
        system_prompt: Instruction for the model role.
        user_prompt:   The actual request/question.
        max_tokens:    Max response length.
        temperature:   Creativity (0=deterministic, 1=creative).
        client:        Optional pre-built client; creates one if None.
    """
    _client = client or get_gemini_client()
    if not _client:
        return ""
    try:
        resp = _client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("GeminiClient: ai_chat failed — %s", exc)
        return ""
