from fastapi import Header, HTTPException, status
from typing import Optional
from . import database


async def require_api_key(x_api_key: Optional[str] = Header(None)) -> dict:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header"
        )

    key_info = await database.validate_key(x_api_key)
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    if not key_info["enabled"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is disabled"
        )

    used, limit = await database.check_rate_limit(x_api_key, key_info["daily_limit"])
    if used > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per day. Used: {used - 1}",
            headers={"X-RateLimit-Limit": str(limit), "X-RateLimit-Used": str(used - 1)}
        )

    return {**key_info, "used_today": used, "daily_limit": limit}
