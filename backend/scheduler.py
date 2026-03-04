"""
scheduler.py — APScheduler-based Task Scheduler
OrchestrAI Autonomous Multi-Agent System

Handles:
  - Scheduling career_agent to run daily at 9:30 AM IST
  - Graceful start/stop
  - Health-check endpoint (optional)
"""

import logging
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

logger = logging.getLogger("CareerAgent.Scheduler")

IST = ZoneInfo("Asia/Kolkata")


# ──────────────────────────────────────────────
# Scheduler factory
# ──────────────────────────────────────────────

def _build_scheduler() -> BackgroundScheduler:
    """Create a BackgroundScheduler configured for IST timezone."""
    scheduler = BackgroundScheduler(timezone=IST)
    return scheduler


def _job_listener(event: JobExecutionEvent) -> None:
    """Log APScheduler job execution events."""
    if event.exception:
        logger.error(
            "Scheduler: Job '%s' FAILED — %s", event.job_id, event.exception
        )
    else:
        logger.info(
            "Scheduler: Job '%s' executed successfully at %s.",
            event.job_id,
            datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S %Z"),
        )


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def schedule_daily_internship_email(job_func, hour: int = 10, minute: int = 45) -> None:
    """
    Schedule `job_func` to run every day at `hour`:`minute` IST.

    Parameters
    ----------
    job_func : callable
        The function to run (e.g., career_agent.run_career_agent).
    hour : int
        Hour in IST (24-hour format). Default 10.
    minute : int
        Minute in IST. Default 45.
    """
    scheduler = _build_scheduler()
    scheduler.add_listener(_job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    trigger = CronTrigger(hour=hour, minute=minute, timezone=IST)
    scheduler.add_job(
        job_func,
        trigger=trigger,
        id="career_agent_daily",
        name="Daily AI Internship Fetch & Email",
        replace_existing=True,
        misfire_grace_time=300,  # allow up to 5 min late execution
    )

    logger.info(
        "Scheduler: Registered 'career_agent_daily' — runs every day at %02d:%02d IST.",
        hour,
        minute,
    )

    # Graceful shutdown on SIGINT / SIGTERM handled natively by Uvicorn now
    # def _shutdown(signum, frame):  # noqa: ANN001
    #     logger.info("Scheduler: Signal %s received — shutting down.", signum)
    #     scheduler.shutdown(wait=False)
    #     sys.exit(0)

    # signal.signal(signal.SIGINT, _shutdown)
    # signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Scheduler: Starting. Press Ctrl+C to stop.")
    scheduler.start()


def run_once_now(job_func) -> None:
    """
    Run `job_func` immediately (useful for manual testing / CI triggers).
    """
    logger.info("Scheduler: Running job immediately (manual trigger).")
    job_func()


# ──────────────────────────────────────────────
# Stand-alone entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    # Allow running: python scheduler.py [--now]
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from backend.agents.execution_agent import run_orchestrai_pipeline  # noqa: E402

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        run_once_now(run_orchestrai_pipeline)
    else:
        schedule_daily_internship_email(run_orchestrai_pipeline)
