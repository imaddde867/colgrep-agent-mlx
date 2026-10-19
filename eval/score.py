#!/usr/bin/env python3
"""Offline, deterministic file/range scoring for official ColGrep agent JSON outputs."""

import argparse
import json
import statistics
from pathlib import Path

from .benchmark import MANIFEST_PATH, digest, manifest

IOU_THRESHOLD = 0.25


def file_name(raw, repo_root):
    if not isinstance(raw, str) or not raw:
        raise ValueError('location has no file path')
    root = Path(repo_root).resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        if '..' in candidate.parts:
            raise ValueError('relative location escapes fixture')
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('location is outside pinned repository')
    return candidate.relative_to(root).as_posix()


def parse_predictions(response, repo_root):
    if not isinstance(response, dict):
        return [], False, None
    submitted = response.get('submitted') is True and response.get('end_reason') == 'Answered' and not response.get('error')
    if not submitted:
        return [], False, None  # inspected fallback is NEVER considered an answer
    predictions = []
    for loc in response.get('locations', []):
        if not isinstance(loc, dict):
            return [], True, 'non-object location'
        try:
            path = file_name(loc.get('file'), repo_root)
            start, end = loc.get('start_line'), loc.get('end_line')
            if type(start) is not int or type(end) is not int or not (0 < start <= end):
                raise ValueError('invalid or absent line range')
            predictions.append((path, start, end))
        except ValueError as exc:
            return [], True, str(exc)  # fail closed, no partial credit for malformed answers
    return list(dict.fromkeys(predictions)), True, None


def overlap(a, b):
    if a[0] != b[0]:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[1], b[1]) + 1)
    union = (a[2] - a[1] + 1) + (b[2] - b[1] + 1) - intersection
    return intersection / union if union else 0.0


def prf(tp, predicted, expected):
    precision = tp / predicted if predicted else 0.0
    recall = tp / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def score_task(task, response, repo_root):
    gold = [(g['file'], g['start'], g['end']) for g in task['gold']]
    predicted, submitted, invalid = parse_predictions(response, repo_root)
    # Every submitted location must be a valid file+range; missing ranges count as failure.
    gold_files = {g[0] for g in gold}
    pred_files = {p[0] for p in predicted}
    file_metrics = prf(len(gold_files & pred_files), len(pred_files), len(gold_files))
    pairs = sorted(
        ((overlap(p, g), pi, gi) for pi, p in enumerate(predicted) for gi, g in enumerate(gold)),
        reverse=True,
    )
    taken_pred, taken_gold = set(), set()
    for iou, pi, gi in pairs:
        if iou < IOU_THRESHOLD:
            break
        if pi not in taken_pred and gi not in taken_gold:
            taken_pred.add(pi)
            taken_gold.add(gi)
    location_metrics = prf(len(taken_pred), len(predicted), len(gold))
    best_gold_ious = [max((overlap(p, g) for p in predicted), default=0.0) for g in gold]
    return {
        'id': task['id'], 'submitted': submitted, 'invalid_location': invalid,
        'predicted_locations': len(predicted), 'expected_locations': len(gold),
        'file': file_metrics, 'location_iou_025': location_metrics,
        'mean_best_gold_iou': statistics.mean(best_gold_ious),
    }


def score_report(data, report):
    if report.get('schema_version') != 1 or report.get('manifest_sha256') != digest(MANIFEST_PATH):
        raise ValueError('Result uses a different manifest; refuse cross-manifest scoring')
    if report.get('fixture_revision') != data['revision'] or report.get('fixture_repo') != data['repository']:
        raise ValueError('Fixture revision/repository mismatch')
    selected = report.get('selected_task_ids')
    all_tasks = {t['id']: t for t in data['tasks']}
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError('Missing/duplicated selected task IDs')
    if any(k not in all_tasks for k in selected):
        raise ValueError('Unknown selected task ID')
    rows_by_id = {}
    for row in report.get('rows', []):
        task_id = row['id']
        if task_id in rows_by_id or task_id not in selected:
            raise ValueError('Unknown/duplicate result row')
        rows_by_id[task_id] = row
    scored = []
    seconds, generated = [], 0
    for task_id in selected:
        row = rows_by_id.get(task_id)
        response = row.get('response') if row else None
        metric = score_task(all_tasks[task_id], response, report['fixture_path'])
        metric['run_error'] = row.get('error') if row else 'missing result'
        if not metric['run_error'] and isinstance(response, dict) and response.get('error'):
            metric['run_error'] = str(response['error'])
        if row and isinstance(row.get('elapsed_seconds'), (float, int)):
            seconds.append(row['elapsed_seconds'])
        if isinstance(response, dict):
            generated += response.get('completion_tokens', 0) or 0
        scored.append(metric)
    macro = lambda getter: round(statistics.mean(getter(x) for x in scored), 6)
    summary = {
        'tasks_selected': len(selected), 'tasks_recorded': len(rows_by_id),
        'complete_manifest': len(selected) == len(data['tasks']) and len(rows_by_id) == len(selected),
        'submitted_rate': macro(lambda x: float(x['submitted'])),
        'file_macro_f1': macro(lambda x: x['file']['f1']),
        'location_macro_f1_iou_025': macro(lambda x: x['location_iou_025']['f1']),
        'mean_best_gold_iou': macro(lambda x: x['mean_best_gold_iou']),
        'invalid_location_tasks': sum(bool(x['invalid_location']) for x in scored),
        'error_tasks': sum(bool(x['run_error']) for x in scored),
        'median_task_seconds': round(statistics.median(seconds), 3) if seconds else None,
        'completion_tokens': generated,
    }
    return {'model_id': report.get('model_id'), 'manifest_sha256': report['manifest_sha256'],
            'fixture_revision': data['revision'], 'summary': summary, 'tasks': scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--compare', type=Path, help='Second report on the same fixture and tasks')
    parser.add_argument('--output', type=Path, help='Optional JSON output (existing file not overwritten)')
    args = parser.parse_args()
    data = manifest()
    report = json.loads(args.report.read_text())
    primary = score_report(data, report)
    result = {'primary': primary}
    if args.compare:
        other = json.loads(args.compare.read_text())
        if other.get('selected_task_ids') != report.get('selected_task_ids'):
            parser.error('Comparisons require exactly the same ordered task IDs')
        if other.get('fixture_revision') != report.get('fixture_revision'):
            parser.error('Fixture revisions differ')
        secondary = score_report(data, other)
        result['comparison'] = secondary
        result['primary_minus_comparison'] = {
            k: round(primary['summary'][k] - secondary['summary'][k], 6)
            for k in ('submitted_rate', 'file_macro_f1', 'location_macro_f1_iou_025', 'mean_best_gold_iou')
        }
    result['warning'] = ('Diagnostic only: 15 hand-labelled public tasks, one stochastic trial; '
                         'partial runs and small differences are not evidence of quality parity.')
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        if args.output.exists():
            parser.error(f'Output already exists: {args.output}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
