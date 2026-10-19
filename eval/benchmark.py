#!/usr/bin/env python3
"""Run the upstream ColGrep agent against a pinned public repository and an MLX server.

The official ColGrep binary owns the agent loop, tools, and read-only terminal.
This script only prepares the fixture, isolates settings, and records results.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / 'eval' / 'tasks.json'
WORK = ROOT / 'work' / 'eval'
RESULTS = ROOT / 'results' / 'eval'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    data = json.loads(MANIFEST_PATH.read_text())
    if data.get('schema_version') != 1 or not re.fullmatch(r'[0-9a-f]{40}', data.get('revision', '')):
        raise ValueError('Unexpected evaluation manifest revision/schema')
    if not data.get('tasks') or len({t['id'] for t in data['tasks']}) != len(data['tasks']):
        raise ValueError('Missing tasks or duplicated task IDs')
    for task in data['tasks']:
        if not re.fullmatch(r'[a-z0-9-]+', task['id']) or not task['question'].strip():
            raise ValueError(f'Invalid task: {task!r}')
        for gold in task['gold']:
            p = Path(gold['file'])
            if p.is_absolute() or '..' in p.parts or not gold['anchor']:
                raise ValueError(f'Unsafe/empty golden location: {gold!r}')
            if not (0 < gold['start'] <= gold['end']):
                raise ValueError(f'Invalid golden range: {gold!r}')
    return data


def checkout_path(data):
    return WORK / 'corpus' / data['slug']


def execute(argv, *, cwd=None, env=None, timeout=900):
    return subprocess.run(argv, cwd=cwd, env=env, text=True,
                          capture_output=True, timeout=timeout, check=False)


def require_ok(proc, argv):
    if proc.returncode:
        raise RuntimeError(f'{argv[0]} failed ({proc.returncode}): {proc.stderr[-3000:]}')


def check_checkout(data):
    path = checkout_path(data)
    if not path.is_dir():
        raise ValueError(f'Fixture absent: {path}; run `python -m eval.benchmark prepare`')
    proc = execute(['git', 'rev-parse', 'HEAD'], cwd=path, timeout=15)
    require_ok(proc, ['git'])
    if proc.stdout.strip() != data['revision']:
        raise ValueError('Fixture revision changed; refusing to benchmark unpinned code')
    proc = execute(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=path, timeout=15)
    require_ok(proc, ['git'])
    if proc.stdout.strip():
        raise ValueError('Fixture has modified or untracked files; refusing to benchmark')
    for task in data['tasks']:
        for item in task['gold']:
            candidate = (path / item['file']).resolve()
            if not candidate.is_relative_to(path.resolve()) or not candidate.is_file():
                raise ValueError(f'Golden file missing: {item["file"]}')
            lines = candidate.read_text().splitlines()
            if item['end'] > len(lines) or item['anchor'] not in '\n'.join(lines[item['start'] - 1:item['end']]):
                raise ValueError(f'Golden anchor not found: {task["id"]}: {item["file"]}')
    return path


def isolated_env(name):
    env = os.environ.copy()
    cfg = WORK / 'config' / name
    data = WORK / 'xdg-data'
    cfg.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    env['XDG_CONFIG_HOME'] = str(cfg.resolve())
    env['XDG_DATA_HOME'] = str(data.resolve())
    return env


def prepare():
    data = manifest()
    dest = checkout_path(data)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        argv = ['git', 'clone', '--no-checkout', '--filter=blob:none', data['repository'], str(dest)]
        require_ok(execute(argv, timeout=600), argv)
        argv = ['git', 'checkout', '--detach', data['revision']]
        require_ok(execute(argv, cwd=dest, timeout=600), argv)
    dest = check_checkout(data)
    env = isolated_env('shared-prepare')
    argv = ['colgrep', 'init', '-y', str(dest)]
    require_ok(execute(argv, env=env, timeout=1200), argv)
    print(f'Prepared pinned {data["slug"]}@{data["revision"]}; {len(data["tasks"])} golden tasks verified')


def endpoint_model(endpoint, model):
    """Fail closed if a running server does not advertise the exact requested model."""
    try:
        with urlopen(endpoint.rstrip('/') + '/models', timeout=10) as response:
            payload = json.load(response)
    except (URLError, OSError, ValueError) as exc:
        raise ValueError(f'MLX server not reachable on {endpoint}: {exc}') from exc
    ids = [m['id'] for m in payload.get('data', []) if isinstance(m, dict) and isinstance(m.get('id'), str)]
    if model not in ids:
        raise ValueError(f'Server does not advertise model {model!r}; available: {ids!r}')
    return model


def configure_colgrep(env, endpoint, model, model_dir):
    harness = ROOT / 'harness'
    argv = [
        'colgrep', 'settings', '--agent-endpoint', endpoint,
        '--agent-endpoint-model', model,
        '--agent-prompt', str(harness / 'system_prompt.txt'),
        '--agent-tools', str(harness / 'tools.json'),
        '--agent-chat-template', str(model_dir / 'chat_template.jinja'),
        '--agent-thinking', 'off', '--agent-max-turns', '10',
        '--agent-search-k', '10', '--agent-max-tokens', '2048',
        '--agent-context', '16384', '--agent-temperature', '0.6',
        '--agent-top-p', '0.95', '--agent-top-k', '20',
        '--agent-seed', '0', '--agent-final-finish', 'on',
    ]
    require_ok(execute(argv, env=env, timeout=30), argv)


def save_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    data = manifest()
    repo = check_checkout(data)
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', args.name):
        raise ValueError('Run name must be alphanumeric with only . _ - separators')
    if not shutil.which('colgrep'):
        raise ValueError('colgrep binary missing; install via `brew install lightonai/tap/colgrep`')
    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / 'chat_template.jinja').is_file():
        raise ValueError('Model directory must contain original chat_template.jinja')
    cfg = json.loads((model_dir / 'config.json').read_text())
    if cfg.get('rope_theta') != 5_000_000:
        raise ValueError('MLX model has wrong top-level rope_theta')
    model_id = endpoint_model(args.endpoint, str(model_dir))
    dest = RESULTS / (args.name + '.json')
    if dest.exists():
        raise ValueError(f'Run already exists: {dest}; use a fresh --name')
    env = isolated_env(args.name)
    configure_colgrep(env, args.endpoint, model_id, model_dir)
    version = execute(['colgrep', '--version'], env=env, timeout=15)
    require_ok(version, ['colgrep'])
    tasks = data['tasks'][:args.limit] if args.limit else data['tasks']
    report = {
        'schema_version': 1, 'manifest_sha256': digest(MANIFEST_PATH),
        'fixture_revision': data['revision'], 'fixture_repo': data['repository'],
        'fixture_path': str(repo.resolve()), 'model_id': model_id,
        'model_config_sha256': digest(model_dir / 'config.json'),
        'chat_template_sha256': digest(model_dir / 'chat_template.jinja'),
        'colgrep_version': version.stdout.strip(), 'endpoint': args.endpoint,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'selected_task_ids': [t['id'] for t in tasks], 'rows': [],
    }
    save_atomic(dest, report)
    for ix, task in enumerate(tasks, 1):
        trace = RESULTS / 'traces' / args.name / (task['id'] + '.json')
        trace.parent.mkdir(parents=True, exist_ok=True)
        sub_env = dict(env, COLGREP_AGENT_TRACE=str(trace.resolve()))
        argv = ['colgrep', '--agent', task['question'], '--json', '--no-update']
        started = time.monotonic()
        stdout_tail = ''
        try:
            result = execute(argv, cwd=repo, env=sub_env, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            raw, error, stderr = None, str(exc), str(exc.stderr or '')[-6000:]
        else:
            stderr = result.stderr[-6000:]
            try:
                raw = json.loads(result.stdout)
                if not isinstance(raw, dict) or raw.get('query') != task['question']:
                    raise ValueError('Expected ColGrep JSON object with original query')
                error = None if result.returncode == 0 else f'colgrep exit {result.returncode}'
            except ValueError as exc:
                raw, error = None, f'Invalid agent output: {exc}'
                stdout_tail = result.stdout[-6000:]
        row = {
            'id': task['id'], 'elapsed_seconds': round(time.monotonic() - started, 3),
            'response': raw, 'error': error, 'stderr_tail': stderr, 'stdout_tail': stdout_tail,
            'trace_sha256': digest(trace) if trace.is_file() else None,
        }
        report['rows'].append(row)
        save_atomic(dest, report)
        status = raw.get('end_reason') if isinstance(raw, dict) else error
        print(f'[{ix}/{len(tasks)}] {task["id"]}: {status} ({row["elapsed_seconds"]:.1f}s)', flush=True)
    print(f'Saved {dest}\nScore: python -m eval.score {dest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare', help='Clone pinned fixture, verify labels, initialize ColGrep search index')
    runner = sub.add_parser('run', help='Run the official agent through a local MLX server')
    runner.add_argument('--name', required=True, help='Unique run identifier, e.g. mlx8-trial1')
    runner.add_argument('--model', required=True, help='Path to local MLX checkpoint directory')
    runner.add_argument('--endpoint', default='http://127.0.0.1:8080/v1')
    runner.add_argument('--limit', type=int, default=0, help='First N tasks only (0 = all)')
    runner.add_argument('--timeout', type=int, default=900, help='Per-task timeout in seconds')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        else:
            if args.limit < 0 or args.timeout < 1:
                parser.error('limit must be >= 0 and timeout must be > 0')
            run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
