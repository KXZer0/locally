"""Service discovery and diagnostics. Locally has no browser frontend."""
from flask import Blueprint
from core.system.status import health, memory_status
from core.system.updates import updates

bp = Blueprint("system", __name__)

@bp.get("/")
def service_info():
    return {"service": "locally", "mode": "headless", "api": "/v1",
            "health": "/health", "models": "/v1/models", "metrics": "/v1/metrics"}

bp.add_url_rule("/health", view_func=health, methods=["GET"])
bp.add_url_rule("/v1/memory", view_func=memory_status, methods=["GET"])
bp.add_url_rule("/v1/updates", view_func=updates, methods=["GET"])
