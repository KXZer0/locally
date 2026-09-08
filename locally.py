#!/usr/bin/env python3
"""Locally: foreground Intel inference service, with optional terminal chat.

python locally.py --device NPU          Run the API for Odysseus or other clients
python locally.py chat                  Attach a terminal to a running server
python locally.py chat --start          Own a server for this chat session
python locally.py status                Inspect a running server
"""
import os
os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("chat", "status"):
        # Client commands need neither Flask nor native OpenVINO imports.
        from core.terminal import main
        raise SystemExit(main(sys.argv[1:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "odysseus":
        # Companion lifecycle needs neither Flask nor native OpenVINO imports.
        from core.odysseus_cli import main
        raise SystemExit(main(sys.argv[2:]))
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "serve":
            del sys.argv[1]
        from core.cli import main
        main()
else:
    from core.app import create_app, create_ollama_app
    app = create_app()
    ollama_app = create_ollama_app()
