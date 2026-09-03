#!/usr/bin/env python3
"""locally — OpenAI-compatible API server for Intel NPU / ARC GPU.

Auto-detects available devices (NPU, GPU, CPU) and model type (VLM/LLM).
NPU-first: works on any Intel Core Ultra laptop. ARC GPU optional.
Dual mode: NPU for chat + GPU for vision, simultaneously.

Usage:
    python locally.py                                        # auto-detect device
    python locally.py --device NPU                           # force NPU
    python locally.py --device GPU                           # force GPU
    python locally.py --gpu-model-dir gpu-model              # dual: NPU chat + GPU vision
    python locally.py --model-dir ~/models/qwen3-14b-int4-ov --device GPU
    python locally.py --whisper-dir whisper-model            # add speech-to-text
    python locally.py --scan                                 # inspect model directories
"""

import os

# This must precede every import which can load OpenVINO.
os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")

from core.app import create_app, create_ollama_app
from core.cli import main


app = create_app()
ollama_app = create_ollama_app()


if __name__ == "__main__":
    main(app, ollama_app)
