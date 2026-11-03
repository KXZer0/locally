from types import SimpleNamespace
import json
from core.tools.native_diagnostics import NativeTrace, repeating_thought, tune_agent_generation


def test_recording_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv('LOCALLY_NATIVE_TRACE_DIR', raising=False)
    trace = NativeTrace(); trace.emit('test', text='private')
    assert trace.path is None


def test_temporary_trace_preserves_raw_text_and_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv('LOCALLY_NATIVE_TRACE_DIR', str(tmp_path))
    trace=NativeTrace(); trace.emit('end', text='<|channel>thought\nraw', token_ids=[1,2], stop_reason='repetition')
    row=json.loads(trace.path.read_text())
    assert row['text']=='<|channel>thought\nraw' and row['token_ids']==[1,2]


def test_only_repeated_thought_is_stopped():
    repeated=list(range(32))*4
    assert repeating_thought(repeated, '<|channel>thought\nrepeated plan')
    assert not repeating_thought(repeated, '<|channel>thought\nplan<channel|>normal answer')
    assert not repeating_thought(list(range(512)), '<|channel>thought\nnonrepeating')
    assert not repeating_thought([1]*64, '<|channel>thought\nshort')


def test_agent_defaults_keep_thinking_bounded_and_use_model_sampling(monkeypatch):
    monkeypatch.delenv('LOCALLY_AGENT_MAX_TOKENS', raising=False)
    gen=SimpleNamespace(max_new_tokens=4096,do_sample=False,temperature=0,top_p=1,top_k=1)
    tune_agent_generation(gen,{})
    assert gen.max_new_tokens==1024 and gen.do_sample and gen.temperature==1.0
    assert gen.top_p==0.95 and gen.top_k==64


def test_explicit_sampling_and_smaller_output_limit_are_preserved(monkeypatch):
    monkeypatch.delenv('LOCALLY_AGENT_MAX_TOKENS', raising=False)
    gen=SimpleNamespace(max_new_tokens=100,do_sample=False,temperature=0,top_p=1,top_k=1)
    tune_agent_generation(gen,{'temperature':0})
    assert gen.max_new_tokens==100 and not gen.do_sample


def test_non_power_of_two_repetition_period():
    assert repeating_thought(list(range(33))*4, '<|channel>thought\nloop')


def test_live_native_guard_stops_before_whole_output_budget():
    import threading
    from unittest.mock import Mock
    from core.slots.device import DeviceSlot
    class Tokenizer:
        def decode(self, ids, **kwargs): return '<|channel>thought\nrepeated planning'
    count = [0]
    class Pipe:
        def get_tokenizer(self): return Tokenizer()
        def generate(self, prompt, generation_config, streamer):
            from openvino_genai import StreamingStatus
            for token in list(range(33))*32:
                count[0] += 1
                if streamer.write(token) == StreamingStatus.STOP: break
            return SimpleNamespace()
    slot=SimpleNamespace(pipe=Pipe(),lock=threading.RLock(),_cancel=threading.Event(),model_name='gemma',_record_turn=Mock())
    import pytest
    with pytest.raises(RuntimeError,match='Reasoning repetition'):
        DeviceSlot.generate_native(slot,'prompt',SimpleNamespace(max_new_tokens=1024,do_sample=True,temperature=1.0,top_p=.95,top_k=64))
    assert count[0] < 200


def test_concise_planning_instruction_is_only_for_thinking_tools():
    from core.tools.gemma import GemmaChat
    chat=GemmaChat("{% for m in messages %}{{m.content}}{% endfor %}")
    messages=[{'role':'system','content':'Original system'},{'role':'user','content':'Task'}]
    assert 'ONE next action' in chat.render(messages,[{'type':'function','function':{'name':'test','parameters':{}}}],True)
    assert 'ONE next action' not in chat.render(messages,[],True)
    assert 'ONE next action' not in chat.render(messages,[{'type':'function','function':{'name':'test','parameters':{}}}],False)
    assert messages[0]['content']=='Original system'
