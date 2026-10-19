import hashlib
import json
import unittest
from convert import patch_for_mlx, card, THETA, ROOT, SOURCE_REVISION


class TestConfig(unittest.TestCase):
    def source(self):
        return {
            'model_type': 'llama', 'hidden_size': 2048,
            'num_hidden_layers': 42, 'num_attention_heads': 16,
            'num_key_value_heads': 2, 'tie_word_embeddings': False,
            'eos_token_id': [1, 130073],
            'rope_parameters': {'rope_type': 'default', 'rope_theta': THETA},
        }

    def test_source_revision_is_pinned(self):
        self.assertRegex(SOURCE_REVISION, r'^[0-9a-f]{40}$')

    def test_rope_fix(self):
        config = self.source()
        result = patch_for_mlx(config, dict(config))
        self.assertEqual(result['rope_theta'], THETA)
        self.assertEqual(result['rope_parameters']['rope_theta'], THETA)

    def test_fail_if_source_rope_differs(self):
        config = self.source()
        config['rope_parameters']['rope_theta'] = 10000
        with self.assertRaises(ValueError):
            patch_for_mlx(config, self.source())

    def test_fail_if_target_rope_conflicts(self):
        config = self.source()
        wrong = dict(config, rope_theta=10000)
        with self.assertRaises(ValueError):
            patch_for_mlx(config, wrong)

    def test_model_card_no_unsupported_quality_claim(self):
        text = card('8bit', 'c' * 40)
        self.assertIn('rope_theta=5000000', text)
        self.assertIn('not claimed to match upstream BF16', text)
        self.assertIn('lightonai/colgrep-agent-2B', text)

    def test_trained_system_prompt_fingerprint(self):
        prompt = (ROOT / 'harness' / 'system_prompt.txt').read_bytes()
        self.assertEqual(hashlib.sha256(prompt).hexdigest(),
            'ea2097d3674ae70d0e0de9447f6d2118af8e091c65b7cbe5ba1c2787223e2897')

    def test_harness_three_tools(self):
        tools = json.loads((ROOT / 'harness' / 'tools.json').read_text())
        self.assertEqual([t['function']['name'] for t in tools], ['colgrep', 'terminal', 'finish'])
        canonical = json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(', ', ': '))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),
            '52ef054c3ab4e6edadbd252c626980cb3ce2204e5ed75f592a411d57d9b12249')


if __name__ == '__main__':
    unittest.main()
