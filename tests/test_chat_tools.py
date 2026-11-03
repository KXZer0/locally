import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from core.tools.activity import record, register, clean_citations
from core.tools.registry import PYTHON_TOOL
from core.chat.turn import _builtin_tool_turn, _prepare_turn
from core.genai.results import GeneratedText

class ChatToolsTests(unittest.TestCase):
    def test_receipts_are_request_scoped_and_memory_only(self):
        app=Flask(__name__);register(app)
        with app.test_request_context(headers={'X-Coucou-Request-Id':'test-request'}):
            record('searching','Searching the web…',query='a question')
        result=app.test_client().get('/v1/chat/activity/test-request').get_json()
        self.assertEqual(result['events'][-1]['query'],'a question')
        self.assertEqual(app.test_client().get('/v1/chat/activity/different').get_json()['events'],[])

    def test_numeric_citations_removed_but_code_untouched(self):
        self.assertEqual(clean_citations('The answer[1] is 391 [2]. `a[1]`'), 'The answer is 391. `a[1]`')

    def test_native_tool_result_and_reasoning_roundtrip(self):
        generated=iter([
            GeneratedText('<|channel>thought\nCalculate.<channel|><|tool_call>call:python{code:<|"|>print(17 * 23)<|"|>}<tool_call|>'),
            GeneratedText('<|channel>thought\nThe result is 391.<channel|>391<turn|>')])
        histories=[]
        def generate(messages):
            histories.append(list(messages));return next(generated)
        with patch.dict('core.chat.turn.BUILTIN_TOOLS',{'python':{'run':lambda args:'391'}}):
            text,_runs=_builtin_tool_turn(generate,[{'role':'user','content':'17*23'}],[PYTHON_TOOL],native=True)
        self.assertEqual(str(text),'391')
        self.assertEqual(text.reasoning,'The result is 391.')
        self.assertEqual(histories[1][-1]['role'],'tool')
        self.assertEqual(histories[1][-1]['content'],'391')
        self.assertEqual(histories[1][-2]['tool_calls'][0]['function']['name'],'python')

    def test_builtin_native_prompt_preserves_original_messages(self):
        slot=MagicMock();slot.model_name='gemma-4-E4B-it';slot.model_type='vlm';slot.device_name='GPU'
        messages=[{'role':'system','content':'Help with math.'},{'role':'user','content':'17*23'}]
        with patch('core.chat.turn.overall_status',return_value='ready'),patch('core.chat.turn._route_request',return_value=slot),patch('core.chat.turn.builtin_tools_for',return_value=[PYTHON_TOOL]):
            turn=_prepare_turn({'messages':messages,'thinking':{'type':'enabled'}},native=True)
        self.assertTrue(turn['native_builtin']);self.assertTrue(turn['thinking'])
        self.assertEqual(turn['raw_messages'],messages)

if __name__=='__main__':unittest.main()
