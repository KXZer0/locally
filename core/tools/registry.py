"""The two tools the server owns, as schemas.

Separate from core/tools/builtin.py, which RUNS them: a schema is answered
before any slot exists, so the registry has to be importable without one."""

# PYTHON_TOOL_ENABLED deliberately does NOT live here. It is config.py's, and
# a second copy in this module was written by nothing and read by builtin.py,
# so --python-tool set the real one while the tool consulted the duplicate and
# stayed off forever. One mutable value, one home.


PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": ("Run a short calculation in local Python. Use this "
                         "for exact arithmetic; do not use it for file or "
                         "network access."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                          "description": "Python code to execute."},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and return source passages. Use it when the answer "
            "depends on current events, on anything past your training data, or "
            "on a specific fact you are not sure of -- check rather than guess. "
            "Do not use it for arithmetic, for reasoning you can do yourself, "
            "or for casual conversation."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "The search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
