#!/usr/bin/env python3
"""First-tool-call smoke test. Does not execute model-generated commands."""

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from convert import ROOT, THETA

CALL_START = '<function'
CALL_END = '</function>'
CDATA_START = '<![CDATA['


class ToolCallError(ValueError):
    """Raised when generated tool-call XML is incomplete or invalid."""


def _call_end(text, start):
    """Return the index just after the matching </function>, skipping CDATA."""
    tag_end = text.find('>', start)
    if tag_end < 0:
        raise ToolCallError('incomplete <function> start tag')
    position = tag_end + 1
    while True:
        close = text.find(CALL_END, position)
        cdata = text.find(CDATA_START, position)
        if close < 0:
            raise ToolCallError('incomplete <function> element: missing </function>')
        if 0 <= cdata < close:
            cdata_end = text.find(']]>', cdata + len(CDATA_START))
            if cdata_end < 0:
                raise ToolCallError('unterminated CDATA section')
            position = cdata_end + 3
            continue
        return close + len(CALL_END)


def _call_blocks(text):
    """Extract complete top-level <function> elements, rejecting truncated ones."""
    blocks = []
    position = 0
    while True:
        start = text.find(CALL_START, position)
        if start < 0:
            break
        boundary = text[start + len(CALL_START):start + len(CALL_START) + 1]
        if not boundary or boundary not in ' \t\r\n>':
            position = start + len(CALL_START)
            continue
        end = _call_end(text, start)
        blocks.append(text[start:end])
        position = end
    if not blocks:
        raise ToolCallError('no complete <function> element found')
    remainder = text
    for block in blocks:
        remainder = remainder.replace(block, '', 1)
    if CALL_END in remainder:
        raise ToolCallError('stray </function> without a matching <function>')
    return blocks


def parse_tool_calls(text, tools):
    """Parse and structurally validate generated tool calls against a tool schema.

    Returns a list of {"name": str, "arguments": {str: str}} dictionaries.
    Raises ToolCallError for truncated, malformed, unknown, or incomplete calls.
    """
    schemas = {tool['function']['name']: tool['function'] for tool in tools}
    blocks = _call_blocks(text)
    try:
        root = ET.fromstring('<tool_calls>' + ''.join(blocks) + '</tool_calls>')
    except ET.ParseError as exc:
        raise ToolCallError(f'tool-call XML is not parseable: {exc}') from exc

    calls = []
    for element in root:
        name = element.get('name')
        if name not in schemas:
            raise ToolCallError(f'unknown tool name: {name!r}')
        parameters = schemas[name]['parameters']
        properties = parameters.get('properties', {})
        arguments = {}
        for param in element:
            if param.tag != 'param':
                raise ToolCallError(f'{name} call contains unexpected <{param.tag}>')
            key = param.get('name')
            if not key:
                raise ToolCallError(f'{name} call has a <param> without a name')
            if key not in properties:
                raise ToolCallError(f'{name} call has unknown argument: {key!r}')
            if key in arguments:
                raise ToolCallError(f'{name} call repeats argument: {key!r}')
            value = ''.join(param.itertext()).strip()
            if properties[key].get('type') == 'integer' and value:
                try:
                    number = int(value)
                except ValueError as exc:
                    raise ToolCallError(f'{name}.{key} is not an integer: {value!r}') from exc
                if number < 1:
                    raise ToolCallError(f'{name}.{key} must be positive')
            arguments[key] = value
        for key in parameters.get('required', []):
            if not arguments.get(key):
                raise ToolCallError(f'{name} call is missing required argument: {key!r}')
        if name == 'colgrep' and not any(
            arguments.get(key, '').strip() for key in ('query', 'pattern', 'command')
        ):
            raise ToolCallError('colgrep call needs a nonempty query, pattern, or command')
        calls.append({'name': name, 'arguments': arguments})
    return calls


def prepare_prompt(tokenizer, question, tools, system):
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': (
            f'Problem statement:\n{question}\n\n'
            'You are at the repository root. Top-level contents:\n'
            'src/  tests/  README.md\n\n'
            'Find the code locations relevant to this problem.'
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        enable_thinking=False, tokenize=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--max-tokens', default=384, type=int)
    args = parser.parse_args()
    model_dir = args.model.resolve()
    card = json.loads((model_dir / 'config.json').read_text())
    if card.get('rope_theta') != THETA:
        parser.error('RoPE theta not fixed in MLX config')
    if card.get('eos_token_id') != [1, 130073]:
        parser.error('EOS IDs missing')
    if not (model_dir / 'chat_template.jinja').is_file():
        parser.error('Original chat_template.jinja missing')

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_dir))
    if model.args.rope_theta != THETA:
        raise RuntimeError(f'Loaded MLX model has incorrect RoPE theta: {model.args.rope_theta}')
    if not getattr(tokenizer, 'chat_template', None):
        raise RuntimeError('Tokenizer lost upstream chat template')

    system = (ROOT / 'harness/system_prompt.txt').read_text()
    tools = json.loads((ROOT / 'harness/tools.json').read_text())
    questions = [
        'Where is retry backoff enforced when an HTTP request fails?',
        'Where does configuration validation reject an empty path?',
    ]
    entries = []
    for question in questions:
        mx.random.seed(0)
        prompt = prepare_prompt(tokenizer, question, tools, system)
        if '<function name=' not in prompt or '<think>\n\n</think>' not in prompt:
            raise RuntimeError('Tool schema or thinking-disabled prompt was not rendered')
        started = time.perf_counter()
        answer = generate(model, tokenizer, prompt=prompt,
                          sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                          max_tokens=args.max_tokens, verbose=False)
        elapsed = time.perf_counter() - started
        validation_error = None
        try:
            calls = parse_tool_calls(answer, tools)
        except ToolCallError as exc:
            calls = []
            validation_error = str(exc)
        names = [call['name'] for call in calls]
        entries.append({
            'question': question,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'generated_text': answer,
            'tool_names': names,
            'valid_first_action': bool(names) and names[0] in {'colgrep', 'terminal'},
            'validation_error': validation_error,
            'wall_seconds': round(elapsed, 3),
        })
        print(f'{question}\n  first tool calls: {names}\n  elapsed: {elapsed:.2f}s\n')
        if validation_error:
            print(f'  validation error: {validation_error}\n')

    path = ROOT / 'results' / f'{model_dir.name}-smoke.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + '\n')
    print(f'Receipt: {path}')
    if not all(x['valid_first_action'] for x in entries):
        raise SystemExit('FAIL: one or more prompts did not produce a valid search/read first tool call')
    print('PASS: model loads with correct RoPE; original tool template renders; first tool calls are well formed.')
    print('This is a smoke test, NOT end-to-end repository retrieval or quality equivalence.')


if __name__ == '__main__':
    main()
