import json
import unittest

from convert import ROOT
from smoke import ToolCallError, parse_tool_calls


class TestToolCalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = json.loads((ROOT / 'harness' / 'tools.json').read_text())

    def parse(self, text):
        return parse_tool_calls(text, self.tools)

    def test_valid_colgrep_query(self):
        text = '<function name="colgrep"><param name="query">retry backoff</param></function>'
        calls = self.parse(text)
        self.assertEqual(calls[0]['name'], 'colgrep')
        self.assertEqual(calls[0]['arguments']['query'], 'retry backoff')

    def test_valid_cdata_argument(self):
        text = (
            '<function name="terminal"><param name="command">'
            '<![CDATA[grep -n "a < b" src/app.py]]>'
            '</param></function>'
        )
        calls = self.parse(text)
        self.assertEqual(calls[0]['arguments']['command'], 'grep -n "a < b" src/app.py')

    def test_valid_finish_call(self):
        text = (
            '<function name="finish"><param name="locations">'
            'src/app.py:10-20\nsrc/config.py:4-8'
            '</param></function>'
        )
        self.assertEqual(self.parse(text)[0]['name'], 'finish')

    def test_incomplete_call(self):
        text = '<function name="colgrep"><param name="query">retry backoff</param>'
        with self.assertRaisesRegex(ToolCallError, 'missing </function>'):
            self.parse(text)

    def test_truncated_start_tag(self):
        with self.assertRaisesRegex(ToolCallError, 'incomplete <function> start tag'):
            self.parse('<function name="colgrep"')

    def test_unknown_tool(self):
        text = '<function name="grep"><param name="query">x</param></function>'
        with self.assertRaisesRegex(ToolCallError, 'unknown tool name'):
            self.parse(text)

    def test_unknown_argument(self):
        text = '<function name="colgrep"><param name="nope">x</param></function>'
        with self.assertRaisesRegex(ToolCallError, 'unknown argument'):
            self.parse(text)

    def test_missing_required_argument(self):
        text = '<function name="terminal"><param name="command"></param></function>'
        with self.assertRaisesRegex(ToolCallError, 'missing required argument'):
            self.parse(text)

    def test_colgrep_requires_usable_argument(self):
        text = '<function name="colgrep"><param name="k">5</param></function>'
        with self.assertRaisesRegex(ToolCallError, 'nonempty query, pattern, or command'):
            self.parse(text)

    def test_malformed_xml(self):
        text = '<function name="colgrep"><param name="query">a & b</param></function>'
        with self.assertRaisesRegex(ToolCallError, 'not parseable'):
            self.parse(text)

    def test_unterminated_cdata(self):
        text = '<function name="terminal"><param name="command"><![CDATA[ls</param></function>'
        with self.assertRaisesRegex(ToolCallError, 'unterminated CDATA'):
            self.parse(text)

    def test_stray_closing_tag(self):
        text = (
            '<function name="terminal"><param name="command">ls</param></function>'
            '</function>'
        )
        with self.assertRaisesRegex(ToolCallError, 'stray </function>'):
            self.parse(text)

    def test_no_call(self):
        with self.assertRaisesRegex(ToolCallError, 'no complete'):
            self.parse('I could not find the relevant code.')


if __name__ == '__main__':
    unittest.main()
