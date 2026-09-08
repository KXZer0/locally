import unittest
from unittest import mock

from core.launch import quick_chat


class ShiftEnterTests(unittest.TestCase):
    def _record(self, key_down=True, key=quick_chat._VK_RETURN, shift=True):
        record = quick_chat._InputRecord()
        record.event_type = quick_chat._KEY_EVENT
        record.event.key_event.key_down = key_down
        record.event.key_event.virtual_key = key
        record.event.key_event.control_key_state = (
            quick_chat._SHIFT_PRESSED if shift else 0)
        return record

    def test_only_shift_enter_triggers_the_shortcut(self):
        self.assertTrue(quick_chat._is_shift_enter(self._record()))
        self.assertFalse(quick_chat._is_shift_enter(self._record(shift=False)))
        self.assertFalse(quick_chat._is_shift_enter(self._record(key=0x41)))
        self.assertFalse(quick_chat._is_shift_enter(self._record(key_down=False)))

    def test_chat_attaches_to_this_api_in_a_new_console(self):
        popen = mock.Mock()
        quick_chat._open_chat(8123, popen=popen)
        command = popen.call_args.args[0]
        self.assertEqual(command[-3:], ["chat", "--url", "http://127.0.0.1:8123"])
        self.assertNotIn("--start", command)
        self.assertEqual(popen.call_args.kwargs["creationflags"],
                         quick_chat._CREATE_NEW_CONSOLE)


if __name__ == "__main__":
    unittest.main()
