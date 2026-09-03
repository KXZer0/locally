"""OpenAI and Anthropic chat routes."""

from flask import Blueprint, jsonify

from core import runtime
from core.chat.anthropic import anthropic_count_tokens, anthropic_messages
from core.chat.openai import chat_completions
from core.chat.ollama import (ollama_chat, ollama_copy, ollama_delete,
                              ollama_generate, ollama_health, ollama_pull,
                              ollama_v1_chat_completions, ollama_version)


bp = Blueprint("chat", __name__)

bp.add_url_rule("/v1/chat/completions", view_func=chat_completions,
                methods=["POST"])
bp.add_url_rule("/v1/messages", view_func=anthropic_messages, methods=["POST"])
bp.add_url_rule("/v1/messages/count_tokens", view_func=anthropic_count_tokens,
                methods=["POST"])


@bp.post("/v1/cancel")
def cancel_generation():
    """Stop any in-progress generation. Returns immediately."""
    for slot in (runtime.primary, runtime.secondary):
        if slot:
            slot.cancel()
    return jsonify({"status": "ok"})


ollama_bp = Blueprint("ollama_chat", __name__)
ollama_bp.add_url_rule("/", view_func=ollama_health, methods=["GET"])
ollama_bp.add_url_rule("/api/version", view_func=ollama_version, methods=["GET"])
ollama_bp.add_url_rule("/api/chat", view_func=ollama_chat, methods=["POST"])
ollama_bp.add_url_rule("/api/generate", view_func=ollama_generate, methods=["POST"])
ollama_bp.add_url_rule("/api/pull", view_func=ollama_pull, methods=["POST"])
ollama_bp.add_url_rule("/api/delete", view_func=ollama_delete, methods=["DELETE"])
ollama_bp.add_url_rule("/api/copy", view_func=ollama_copy, methods=["POST"])
ollama_bp.add_url_rule("/v1/chat/completions",
                       view_func=ollama_v1_chat_completions, methods=["POST"])
