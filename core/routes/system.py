"""Root, diagnostics, setup, and companion-process routes."""

from flask import Blueprint

from core.system.browser import launch_claude_code, open_external
from core.system.coding import (coding_mode, opencode_status,
                                opencode_web_start, opencode_web_stop)
from core.system.odysseus_api import (launch_opencode, odysseus_set_autostart,
                                      odysseus_start, odysseus_status,
                                      odysseus_stop)
from core.system.setup import (setup_finish, setup_install,
                               setup_install_status, setup_state)
from core.system.status import gui, health, memory_status, ui_bootstrap
from core.system.updates import updates


bp = Blueprint("system", __name__)

bp.add_url_rule("/", view_func=gui, methods=["GET"])
bp.add_url_rule("/health", view_func=health, methods=["GET"])
bp.add_url_rule("/v1/memory", view_func=memory_status, methods=["GET"])
bp.add_url_rule("/v1/ui/bootstrap", view_func=ui_bootstrap, methods=["GET"])
bp.add_url_rule("/v1/updates", view_func=updates, methods=["GET"])
bp.add_url_rule("/v1/setup", view_func=setup_state, methods=["GET"])
bp.add_url_rule("/v1/setup/install", view_func=setup_install, methods=["POST"])
bp.add_url_rule("/v1/setup/install/<job_id>", view_func=setup_install_status,
                methods=["GET"])
bp.add_url_rule("/v1/setup/finish", view_func=setup_finish, methods=["POST"])
bp.add_url_rule("/v1/open", view_func=open_external, methods=["POST"])
bp.add_url_rule("/v1/code/launch", view_func=launch_claude_code,
                methods=["POST"])
bp.add_url_rule("/v1/coding-mode", view_func=coding_mode,
                methods=["GET", "POST"])
bp.add_url_rule("/v1/opencode/status", view_func=opencode_status,
                methods=["GET"])
bp.add_url_rule("/v1/opencode/web", view_func=opencode_web_start,
                methods=["POST"])
bp.add_url_rule("/v1/opencode/web", view_func=opencode_web_stop,
                methods=["DELETE"])
bp.add_url_rule("/v1/opencode/launch", view_func=launch_opencode,
                methods=["POST"])
bp.add_url_rule("/v1/odysseus", view_func=odysseus_status, methods=["GET"])
bp.add_url_rule("/v1/odysseus/start", view_func=odysseus_start,
                methods=["POST"])
bp.add_url_rule("/v1/odysseus/autostart", view_func=odysseus_set_autostart,
                methods=["POST"])
bp.add_url_rule("/v1/odysseus/stop", view_func=odysseus_stop,
                methods=["POST"])
