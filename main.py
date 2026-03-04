"""
main.py — OrchestrAI Career Agent Entry Point
Runs on Render as a FastAPI Web Service.

Flow:
  Local .env  →  Local YAML DB  →  FastAPI + Scheduler  →  Gmail email
"""

import logging
import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("OrchestrAI")

# ── Validate required env vars before starting ────────────────────────────────
REQUIRED_VARS = [
    "EMAIL_USER",
    "EMAIL_PASS",
    "EMAIL_RECEIVER",
]

def _check_env() -> None:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Set them in your .env file or Render Environment.")
        sys.exit(1)
    logger.info("All required environment variables are present ✓")

# ── Entry point ───────────────────────────────────────────────────────────────
def run_full_pipeline():
    from backend.agents.execution_agent import run_orchestrai_pipeline
    run_orchestrai_pipeline()

_check_env()

# ── FastAPI Web Server Setup ───────────────────────────────────────────────
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="OrchestrAI Dashboard", description="Hosts locally generated application files.")
DATA_DIR = os.getenv("DATA_DIR", ".")

# Ensure directories exist
try:
    for directory in ["database", "application_packages", "frontend/practice", "frontend/portfolio", "optimized_resumes", "cover_letters"]:
        os.makedirs(os.path.join(DATA_DIR, directory), exist_ok=True)
except PermissionError:
    logger.warning(f"Permission denied writing to DATA_DIR '{DATA_DIR}'. Falling back to local './data' directory.")
    DATA_DIR = "./data"
    for directory in ["database", "application_packages", "frontend/practice", "frontend/portfolio", "optimized_resumes", "cover_letters"]:
        os.makedirs(os.path.join(DATA_DIR, directory), exist_ok=True)

app.mount("/database", StaticFiles(directory=os.path.join(DATA_DIR, "database")), name="database")
app.mount("/application_packages", StaticFiles(directory=os.path.join(DATA_DIR, "application_packages")), name="application_packages")
app.mount("/frontend/practice", StaticFiles(directory=os.path.join(DATA_DIR, "frontend/practice")), name="practice")
app.mount("/portfolio", StaticFiles(directory=os.path.join(DATA_DIR, "frontend/portfolio"), html=True), name="portfolio")
app.mount("/optimized_resumes", StaticFiles(directory=os.path.join(DATA_DIR, "optimized_resumes")), name="optimized_resumes")
app.mount("/cover_letters", StaticFiles(directory=os.path.join(DATA_DIR, "cover_letters")), name="cover_letters")

@app.on_event("startup")
def start_scheduler():
    from backend.scheduler import schedule_daily_internship_email, run_once_now
    
    if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
        logger.info("Startup trigger: running full pipeline immediately in background...")
        t = threading.Thread(target=run_once_now, args=(run_full_pipeline,))
        t.start()
        
    logger.info("Starting scheduler: full pipeline will run daily at 9:30 AM IST.")
    schedule_daily_internship_email(run_full_pipeline, hour=9, minute=30)
    
@app.get("/", response_class=HTMLResponse)
def index():
    return "<h1>OrchestrAI Hosted on Render 🚀</h1><p>Agent static files are ready.</p><p><a href='/trigger'>Click here to manually trigger the email pipeline</a></p>"

@app.get("/trigger", response_class=HTMLResponse)
def trigger_pipeline(background_tasks: BackgroundTasks):
    logger.info("Manual HTTP trigger: running full pipeline in background...")
    background_tasks.add_task(run_full_pipeline)
    return "<h1>Pipeline Triggered! 🚀</h1><p>The AI agents are running in the background. Please check your Render logs! The email will arrive in your inbox in 5-10 minutes once all 100+ jobs and portfolios are processed.</p>"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        logger.info("Manual trigger: running full pipeline immediately...")
        from backend.scheduler import run_once_now
        run_once_now(run_full_pipeline)
    else:
        import uvicorn
        port = int(os.environ.get("PORT", "10000"))
        logger.info(f"Main block executing: Starting Uvicorn on 0.0.0.0:{port}")
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
