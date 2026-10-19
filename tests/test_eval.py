import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval import benchmark
from eval.score import IOU_THRESHOLD, file_name, overlap, parse_predictions, score_report, score_task


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'src').mkdir()
        self.task = {'id': 'test', 'gold': [{'file': 'src/a.py', 'start': 10, 'end': 20}]}

    def response(self, *, submitted=True, reason='Answered', locs=None):
        return {'submitted': submitted, 'end_reason': reason, 'error': None,
                'locations': locs if locs is not None else [
                    {'file': str(self.root / 'src/a.py'), 'start_line': 10, 'end_line': 20}]}

    def test_manifest_has_pinned_tasks(self):
        m = benchmark.manifest()
        self.assertEqual(m['revision'], '611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60')
        self.assertEqual(len(m['tasks']), 15)

    def test_perfect_exact_location(self):
        scored = score_task(self.task, self.response(), self.root)
        self.assertEqual(scored['file']['f1'], 1)
        self.assertEqual(scored['location_iou_025']['f1'], 1)
        self.assertEqual(scored['mean_best_gold_iou'], 1)

    def test_filename_correct_range_wrong(self):
        loc = {'file': 'src/a.py', 'start_line': 100, 'end_line': 105}
        scored = score_task(self.task, self.response(locs=[loc]), self.root)
        self.assertEqual(scored['file']['f1'], 1)
        self.assertEqual(scored['location_iou_025']['f1'], 0)

    def test_inspected_fallback_never_scores(self):
        scored = score_task(self.task, self.response(submitted=False, reason='MaxTurns'), self.root)
        self.assertFalse(scored['submitted'])
        self.assertEqual(scored['file']['f1'], 0)

    def test_invalid_path_cannot_receive_credit(self):
        outside = self.root.parent / 'outside.py'
        scored = score_task(self.task, self.response(locs=[
            {'file': 'src/a.py', 'start_line': 10, 'end_line': 20},
            {'file': str(outside), 'start_line': 10, 'end_line': 20},
        ]), self.root)
        self.assertIn('outside', scored['invalid_location'])
        self.assertEqual(scored['file']['f1'], 0)

    def test_bad_or_absent_line_ranges_cannot_score(self):
        for loc in [
            {'file': 'src/a.py', 'start_line': None, 'end_line': None},
            {'file': 'src/a.py', 'start_line': 1, 'end_line': 0},
            {'file': 'src/a.py', 'start_line': True, 'end_line': 3},
        ]:
            with self.subTest(loc=loc):
                self.assertEqual(score_task(self.task, self.response(locs=[loc]), self.root)['file']['f1'], 0)

    def test_relative_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            file_name('../a.py', self.root)

    def test_deduplicate_predictions(self):
        loc = {'file': 'src/a.py', 'start_line': 10, 'end_line': 20}
        pred, submitted, err = parse_predictions(self.response(locs=[loc, loc]), self.root)
        self.assertEqual(len(pred), 1)
        self.assertTrue(submitted)
        self.assertIsNone(err)

    def test_iou_uses_inclusive_boundaries(self):
        self.assertEqual(overlap(('f', 1, 1), ('f', 1, 1)), 1)
        self.assertEqual(overlap(('f', 1, 3), ('f', 3, 5)), 1 / 5)
        self.assertEqual(overlap(('f', 1, 3), ('z', 1, 3)), 0)
        self.assertEqual(IOU_THRESHOLD, 0.25)

    def test_result_manifest_hash_must_match(self):
        m = benchmark.manifest()
        with self.assertRaisesRegex(ValueError, 'different manifest'):
            score_report(m, {'schema_version': 1, 'manifest_sha256': 'bad'})

    def test_partial_report_explicitly_not_complete(self):
        m = benchmark.manifest()
        r = {'schema_version': 1, 'manifest_sha256': benchmark.digest(benchmark.MANIFEST_PATH),
             'fixture_revision': m['revision'], 'fixture_repo': m['repository'],
             'fixture_path': str(self.root), 'model_id': 'candidate',
             'selected_task_ids': [m['tasks'][0]['id']], 'rows': []}
        scored = score_report(m, r)
        self.assertEqual(scored['summary']['tasks_selected'], 1)
        self.assertFalse(scored['summary']['complete_manifest'])
        self.assertEqual(scored['summary']['file_macro_f1'], 0)

    def test_unknown_result_rows_rejected(self):
        m = benchmark.manifest()
        r = {'schema_version': 1, 'manifest_sha256': benchmark.digest(benchmark.MANIFEST_PATH),
             'fixture_revision': m['revision'], 'fixture_repo': m['repository'],
             'fixture_path': str(self.root), 'model_id': 'candidate',
             'selected_task_ids': [m['tasks'][0]['id']],
             'rows': [{'id': 'not-in-manifest'}]}
        with self.assertRaisesRegex(ValueError, 'Unknown/duplicate'):
            score_report(m, r)


if __name__ == '__main__':
    unittest.main()
