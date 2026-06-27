import asyncio
import logging
import os
import time
from typing import Optional

from . import database

logger = logging.getLogger(__name__)

_queue: asyncio.Queue = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None


async def enqueue(job_id: str, job_func, *args, **kwargs):
    await _queue.put((job_id, job_func, args, kwargs))
    logger.info(f"Job {job_id} enqueued (queue size: {_queue.qsize()})")


async def _worker():
    logger.info("Job queue worker started.")
    while True:
        try:
            item = await _queue.get()
            job_id, job_func, args, kwargs = item
            logger.info(f"Processing job {job_id}...")

            try:
                await job_func(job_id, *args, **kwargs)
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)
                await database.update_job_status(job_id, "failed", error=str(e))
            finally:
                _queue.task_done()
        except asyncio.CancelledError:
            logger.info("Job queue worker cancelled.")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)


def start_worker():
    global _worker_task
    _worker_task = asyncio.create_task(_worker())
    return _worker_task


def stop_worker():
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()


def queue_size() -> int:
    return _queue.qsize()
