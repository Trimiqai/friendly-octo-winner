import sqlite3
import secrets
import asyncio
from datetime import date, datetime
from typing import Optional
from functools import partial
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upscaler.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                daily_limit INTEGER NOT NULL DEFAULT 100
            );

            CREATE TABLE IF NOT EXISTS api_usage (
                key TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key, usage_date)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                job_type TEXT NOT NULL DEFAULT 'image',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                output_file TEXT,
                error TEXT,
                scale INTEGER NOT NULL DEFAULT 4,
                fmt TEXT NOT NULL DEFAULT 'PNG',
                compression TEXT NOT NULL DEFAULT 'balanced'
            );
        """)


async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args))


def _generate_api_key(name: str = "") -> dict:
    key = "resr_" + secrets.token_urlsafe(32)
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys (key, name, created_at, enabled, daily_limit) VALUES (?, ?, ?, 1, 100)",
            (key, name, now)
        )
    return {"key": key, "name": name, "created_at": now, "enabled": True, "daily_limit": 100}


def _list_api_keys() -> list:
    with _get_conn() as conn:
        rows = conn.execute("SELECT key, name, created_at, enabled, daily_limit FROM api_keys ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def _delete_api_key(key: str) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM api_keys WHERE key = ?", (key,))
    return cur.rowcount > 0


def _set_key_enabled(key: str, enabled: bool) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("UPDATE api_keys SET enabled = ? WHERE key = ?", (1 if enabled else 0, key))
    return cur.rowcount > 0


def _validate_key(key: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT key, enabled, daily_limit FROM api_keys WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def _check_rate_limit(key: str, daily_limit: int) -> tuple[int, int]:
    today = date.today().isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT count FROM api_usage WHERE key = ? AND usage_date = ?", (key, today)
        ).fetchone()
        count = row["count"] if row else 0
        if count >= daily_limit:
            return count, daily_limit
        conn.execute(
            "INSERT INTO api_usage (key, usage_date, count) VALUES (?, ?, 1) "
            "ON CONFLICT(key, usage_date) DO UPDATE SET count = count + 1",
            (key, today)
        )
    return count + 1, daily_limit


def _get_usage_for_key(key: str) -> dict:
    today = date.today().isoformat()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT count FROM api_usage WHERE key = ? AND usage_date = ?", (key, today)
        ).fetchone()
    return {"used_today": row["count"] if row else 0, "date": today}


def _create_job(job_id: str, api_key: str, job_type: str, scale: int, fmt: str, compression: str) -> dict:
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, api_key, status, job_type, created_at, updated_at, scale, fmt, compression) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)",
            (job_id, api_key, job_type, now, now, scale, fmt, compression)
        )
    return {"job_id": job_id, "status": "queued", "created_at": now}


def _update_job_status(job_id: str, status: str, output_file: str = None, error: str = None):
    now = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, output_file = ?, error = ? WHERE job_id = ?",
            (status, now, output_file, error, job_id)
        )


def _get_job(job_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def _cleanup_old_jobs(hours: int = 24):
    cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT job_id, output_file FROM jobs WHERE created_at < ? AND status IN ('completed', 'failed')",
            (cutoff,)
        ).fetchall()
        for row in rows:
            if row["output_file"] and os.path.exists(row["output_file"]):
                try:
                    os.remove(row["output_file"])
                except OSError:
                    pass
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))


async def init_db():
    await run_in_executor(_init_db)


async def generate_api_key(name: str = "") -> dict:
    return await run_in_executor(_generate_api_key, name)


async def list_api_keys() -> list:
    return await run_in_executor(_list_api_keys)


async def delete_api_key(key: str) -> bool:
    return await run_in_executor(_delete_api_key, key)


async def set_key_enabled(key: str, enabled: bool) -> bool:
    return await run_in_executor(_set_key_enabled, key, enabled)


async def validate_key(key: str) -> Optional[dict]:
    return await run_in_executor(_validate_key, key)


async def check_rate_limit(key: str, daily_limit: int) -> tuple[int, int]:
    return await run_in_executor(_check_rate_limit, key, daily_limit)


async def get_usage_for_key(key: str) -> dict:
    return await run_in_executor(_get_usage_for_key, key)


async def create_job(job_id: str, api_key: str, job_type: str, scale: int, fmt: str, compression: str) -> dict:
    return await run_in_executor(_create_job, job_id, api_key, job_type, scale, fmt, compression)


async def update_job_status(job_id: str, status: str, output_file: str = None, error: str = None):
    await run_in_executor(_update_job_status, job_id, status, output_file, error)


async def get_job(job_id: str) -> Optional[dict]:
    return await run_in_executor(_get_job, job_id)


async def cleanup_old_jobs():
    await run_in_executor(_cleanup_old_jobs)
