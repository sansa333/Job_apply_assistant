from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import settings
from app.domestic.service import DomesticJobService


logger = logging.getLogger(__name__)
_scheduler: BackgroundScheduler | None = None


def _scheduled_sync() -> None:
    try:
        service = DomesticJobService()
        if settings.domestic_sync_build_index:
            service.refresh_all()
        else:
            service.sync_all(build_index=False)
    except Exception:
        logger.exception("Domestic job synchronization failed")


def start_domestic_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not settings.domestic_sync_enabled:
        return None
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
    scheduler.add_job(
        _scheduled_sync,
        trigger="interval",
        minutes=max(15, settings.domestic_sync_interval_minutes),
        id="domestic_official_job_sync",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()
    _scheduler = scheduler
    return scheduler


def stop_domestic_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
