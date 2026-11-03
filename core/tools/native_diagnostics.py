"""Bounded native Gemma reasoning and explicitly opt-in diagnostic traces."""
import json
import os
import time
import uuid
from pathlib import Path


def tune_agent_generation(gen, body):
    """Only called for native Gemma thinking+tool turns, never dictation."""
    limit = int(os.environ.get('LOCALLY_AGENT_MAX_TOKENS', '1024'))
    if limit > 0:
        gen.max_new_tokens = min(gen.max_new_tokens, limit)
    # Match the downloaded Gemma generation_config when the caller omits sampling.
    if body.get('temperature') is None:
        gen.do_sample = True
        gen.temperature = 1.0
        gen.top_p = body.get('top_p', 0.95)
        gen.top_k = body.get('top_k', 64)


def repeating_thought(ids, text):
    """Four verbatim long token blocks inside an OPEN thought channel only."""
    if len(ids) < 128 or text.rfind('<|channel>') <= text.rfind('<channel|>'):
        return False
    for width in range(16, min(128, len(ids) // 4) + 1):
        if len(ids) >= width * 4:
            tail = ids[-width * 4:]
            if all(tail[i * width:(i + 1) * width] == tail[:width] for i in range(1, 4)):
                return True
    return False


class NativeTrace:
    """No files or recording unless a temporary launch explicitly opts in."""
    def __init__(self):
        self.path = None
        root = os.environ.get('LOCALLY_NATIVE_TRACE_DIR')
        if root:
            try:
                p = Path(root); p.mkdir(parents=True, exist_ok=True)
                self.path = p / (uuid.uuid4().hex + '.jsonl')
            except OSError:
                pass

    def emit(self, event, **data):
        if self.path is None:
            return
        try:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'time': time.time(), 'event': event, **data}, ensure_ascii=False) + '\n')
        except OSError:
            self.path = None
