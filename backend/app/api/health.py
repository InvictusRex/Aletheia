"""Health endpoint.

Distinguishes application health from dependency health: the endpoint
always returns HTTP 200 when the app is running, with a nested
``database`` object reporting whether PostgreSQL is reachable.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.db.session import check_database_connection

router = APIRouter()


@router.get("/health")
def health() -> dict:
    reachable, detail = check_database_connection()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
        "database": {
            "configured": bool(settings.database_url),
            "reachable": reachable,
            "detail": detail,
        },
    }
