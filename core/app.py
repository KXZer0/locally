"""Flask application factories for the OpenAI and Ollama surfaces."""

import os

from flask import Flask

from core import config
from core.metrics import register_metrics_route
from core.routes import audio
from core.routes.chat import bp as chat_bp, ollama_bp as ollama_chat_bp
from core.routes.models import bp as models_bp, ollama_bp as ollama_models_bp
from core.routes.system import bp as system_bp
from core.routes.util import bp as util_bp
from core.system.status import _log_request


def create_app():
    app = Flask("locally",
                template_folder=os.path.join(config.SCRIPT_DIR, "templates"),
                static_folder=os.path.join(config.SCRIPT_DIR, "static"))
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.register_blueprint(chat_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(audio.bp)
    app.register_blueprint(util_bp)
    app.register_blueprint(system_bp)
    # /v1/metrics is registered rather than blueprinted: it owns a single
    # route over a ring buffer and has no surface of its own.
    register_metrics_route(app)

    @app.before_request
    def debug_openai():
        _log_request("OpenAI")

    audio.attach_socket(app)
    return app


def create_ollama_app():
    app = Flask("locally-Ollama")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_REQUEST_BYTES
    app.register_blueprint(ollama_chat_bp)
    app.register_blueprint(ollama_models_bp)

    @app.before_request
    def debug_ollama():
        _log_request("Ollama")

    return app
