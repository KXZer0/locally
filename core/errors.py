"""The two error shapes every request path shares.

Here rather than in locally.py so that a leaf module -- the web reader
refusing a private address, say -- can reject a request without importing
the server, which would be a cycle.
"""
from flask import jsonify


def openai_error(message, error_type="invalid_request_error", status=400):
    return jsonify({"error": {"message": message, "type": error_type}}), status


class _TurnError(Exception):
    """Carries a ready-made error Response out of the shared setup path."""

    def __init__(self, response):
        super().__init__("turn setup failed")
        self.response = response
