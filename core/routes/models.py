"""Model discovery and lifecycle routes."""

from flask import Blueprint, jsonify

from core.models.discovery import _available_models_data, _models_data
from core.models.manage import load_model, unload_model
from core.chat.ollama import ollama_show, ollama_tags


bp = Blueprint("models", __name__)


@bp.get("/v1/models")
def list_models():
    return jsonify(_models_data())


@bp.get("/v1/models/available")
def list_available_models():
    return jsonify(_available_models_data())


bp.add_url_rule("/v1/models/load", view_func=load_model, methods=["POST"])
bp.add_url_rule("/v1/models/unload", view_func=unload_model, methods=["POST"])

ollama_bp = Blueprint("ollama_models", __name__)
ollama_bp.add_url_rule("/api/tags", view_func=ollama_tags, methods=["GET"])
ollama_bp.add_url_rule("/api/show", view_func=ollama_show, methods=["POST"])
