"""Document, image, calculation, and search utility routes."""

from flask import Blueprint

from core.util.images import (util_background, util_detect, util_generate,
                              util_upscale)
from core.util.python import util_python
from core.util.read import util_read
from core.util.search import util_index, util_search
from core.web.api import read_url, web_search


bp = Blueprint("util", __name__)

bp.add_url_rule("/v1/util/read", view_func=util_read, methods=["POST"])
bp.add_url_rule("/v1/util/background", view_func=util_background, methods=["POST"])
bp.add_url_rule("/v1/util/upscale", view_func=util_upscale, methods=["POST"])
bp.add_url_rule("/v1/util/detect", view_func=util_detect, methods=["POST"])
bp.add_url_rule("/v1/util/python", view_func=util_python, methods=["POST"])
bp.add_url_rule("/v1/util/generate", view_func=util_generate, methods=["POST"])
bp.add_url_rule("/v1/util/index", view_func=util_index, methods=["POST"])
bp.add_url_rule("/v1/util/search", view_func=util_search, methods=["POST"])
bp.add_url_rule("/v1/read-url", view_func=read_url, methods=["POST"])
bp.add_url_rule("/v1/search", view_func=web_search, methods=["POST"])
