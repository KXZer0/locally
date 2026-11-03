"""Short-lived tool receipts for the active Coucou request; memory only."""
import re
import threading
import time
from collections import OrderedDict
from flask import has_request_context, request, jsonify

_lock = threading.Lock()
_runs = OrderedDict()

def clean_citations(text):
    # Keep fenced/inline code intact; only remove rendered prose citation labels.
    chunks = re.split(r'(```[\s\S]*?```|`[^`]*`)', text)
    for i in range(0, len(chunks), 2):
        chunks[i] = re.sub(r'\s*\[\d+(?:\s*[,;-]\s*\d+)*\](?:\(https?://[^)]+\))?', '', chunks[i])
    return ''.join(chunks)

def request_id():
    if not has_request_context():
        return ''
    value = request.headers.get('X-Coucou-Request-Id', '')
    return value if re.fullmatch(r'[a-zA-Z0-9-]{1,80}', value) else ''

def record(phase, text, **details):
    key = request_id()
    if not key:
        return
    with _lock:
        now = time.monotonic()
        for old in list(_runs):
            if now - _runs[old]['at'] > 180:
                del _runs[old]
        run = _runs.setdefault(key, {'at':now,'events':[],'sequence':0})
        run['at'] = now
        run['sequence'] += 1
        run['events'].append({'seq':run['sequence'],'phase':phase,'text':str(text)[:400],**details})
        run['events'] = run['events'][-60:]
        while len(_runs) > 32:
            _runs.popitem(last=False)

def register(app):
    @app.get('/v1/chat/activity/<key>')
    def activity(key):
        if not re.fullmatch(r'[a-zA-Z0-9-]{1,80}', key):
            return jsonify({'events':[]}), 400
        with _lock:
            run = _runs.get(key)
            events = list(run['events']) if run and time.monotonic()-run['at'] < 180 else []
        return jsonify({'events':events})
