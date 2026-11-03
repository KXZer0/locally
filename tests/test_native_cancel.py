import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from core.slots.device import DeviceSlot
from core.chat.openai import _native_response


def test_cancelled_native_request_never_enters_pipeline():
    event = threading.Event(); event.set()
    slot = SimpleNamespace(pipe=Mock())
    with pytest.raises(RuntimeError, match='cancelled'):
        DeviceSlot.generate_native(slot, 'test', SimpleNamespace(), cancel_event=event)
    slot.pipe.generate.assert_not_called()


def test_queued_native_request_rechecks_its_own_cancel_before_clearing_global():
    event = threading.Event(); global_cancel = threading.Event(); global_cancel.set()
    class Lock:
        def __enter__(self): event.set()
        def __exit__(self, *args): pass
    slot = SimpleNamespace(pipe=Mock(), lock=Lock(), _cancel=global_cancel)
    with pytest.raises(RuntimeError, match='cancelled'):
        DeviceSlot.generate_native(slot, 'test', SimpleNamespace(), cancel_event=event)
    slot.pipe.generate.assert_not_called()
    assert global_cancel.is_set()


def test_closing_native_stream_signals_only_that_request():
    captured = {}
    def generate(prompt, gen, cancel_event=None):
        captured['event'] = cancel_event
        return 'ok'
    slot = SimpleNamespace(generate_native=generate, cancel=Mock())
    def frames(*args, generate=None, **kwargs):
        generate()
        yield 'data: heartbeat\n\n'
        yield 'data: done\n\n'
    turn = {'slot':slot, 'gen':SimpleNamespace(), 'tools':[], 'native_prompt':'prompt', 'stream':True}
    with patch('core.chat.openai._sse_tool_stream', frames), patch('core.chat.openai._turn_headers', return_value={}):
        response = _native_response(turn, 'test', 0, 0)
        iterator = iter(response.response)
        next(iterator)
        assert not captured['event'].is_set()
        response.close()
        assert captured['event'].is_set()
        slot.cancel.assert_not_called()
