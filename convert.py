#!/usr/bin/env python3
"""Pinned ColGrep-Agent-2B -> MLX conversion with validated RoPE compatibility."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

SOURCE = 'lightonai/colgrep-agent-2B'
SOURCE_REVISION = 'f798a61129c6decb5166864e0a3c98373e62c764'
THETA = 5_000_000
ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(config):
    expected = {
        'model_type': 'llama',
        'hidden_size': 2048,
        'num_hidden_layers': 42,
        'num_attention_heads': 16,
        'num_key_value_heads': 2,
        'tie_word_embeddings': False,
        'eos_token_id': [1, 130073],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Source changed: {key}: {config.get(key)!r} != {value!r}')
    actual_theta = (config.get('rope_parameters') or {}).get('rope_theta')
    if actual_theta != THETA:
        raise ValueError(f'Source RoPE changed: {actual_theta!r} != {THETA}')


def patch_for_mlx(source_config, target_config):
    """Restore the source RoPE theta where MLX-LM 0.32.0 expects it.

    The source stores RoPE under rope_parameters.rope_theta, while MLX-LM's
    Llama class reads a top-level rope_theta and otherwise falls back to 10000.
    This adds the flat compatibility key only; no model weights are altered.
    """
    verify_source(source_config)
    if target_config.get('model_type') != 'llama':
        raise ValueError(f'Unexpected target architecture: {target_config.get("model_type")!r}')
    old = target_config.get('rope_theta')
    if old is not None and old != THETA:
        raise ValueError(f'Conflicting top-level RoPE theta: {old}')
    target_config['rope_theta'] = THETA
    if target_config.get('eos_token_id') != [1, 130073]:
        raise ValueError(f'Target EOS token IDs changed: {target_config.get("eos_token_id")!r}')
    return target_config


def card(variant, source_sha):
    label = '8-bit affine, group size 64' if variant == '8bit' else 'BF16 reference (no quantization)'
    return f'''---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: lightonai/colgrep-agent-2B
tags:
- mlx
- code-search
- tool-use
- apple-silicon
---

# ColGrep-Agent-2B - MLX {variant.upper()}

Native Apple Silicon MLX conversion of [LightOn's ColGrep-Agent-2B](https://huggingface.co/lightonai/colgrep-agent-2B), a specialized repository code-localization agent built on MiniCPM5-2B.

- Upstream checkpoint: `{SOURCE}` at `{source_sha}`
- Conversion: MLX-LM 0.32.0, {label}
- Compatibility: source `chat_template.jinja` preserved; top-level `rope_theta=5000000` added so MLX-LM's Llama RoPE implementation matches the source configuration.
- Original model authors: LightOn AI; original model license: Apache-2.0.

## Usage

```bash
pip install 'mlx-lm==0.32.0' 'transformers>=5.6.2,<6'
mlx_lm.generate --model . --prompt 'Hello' --max-tokens 128
```

**This is not a general coding assistant.** It was trained for iterative code search with `colgrep`, `terminal`, and `finish`. Use the upstream chat template, system prompt, tool schemas, and thinking disabled. The official `colgrep --agent` CLI currently runs GGUF through llama.cpp; this MLX checkpoint is **not** a drop-in replacement for that CLI. An MLX-backed harness is separate integration work.

This checkpoint is not claimed to match upstream BF16 on benchmarks until an apples-to-apples evaluation of full search trajectories is reported. The smoke test validates basic load, format, and first tool-call structure only.

- Upstream blog: https://huggingface.co/blog/lightonai/colgrep-agent
- Upstream harness: https://github.com/lightonai/next-plaid
- MLX-LM: https://github.com/ml-explore/mlx-lm
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['8bit', 'bf16'], default='8bit')
    parser.add_argument(
        '--revision', default=SOURCE_REVISION,
        help=f'Upstream commit SHA (default: verified revision {SOURCE_REVISION})',
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    revision = args.revision
    if len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
        raise SystemExit(f'Expected a 40-character lowercase upstream commit SHA, got {revision!r}')

    upstream = ROOT / 'work' / 'source' / revision
    output = ROOT / 'release' / f'ColGrep-Agent-2B-MLX-{args.variant}'
    if output.exists():
        parser.error(f'{output} already exists; no automatic overwrite')
    print(f'[SOURCE] {SOURCE}@{revision}', flush=True)
    snapshot_download(
        repo_id=SOURCE, revision=revision, local_dir=str(upstream),
        allow_patterns=['*.json', '*.jinja', '*.safetensors', 'README.md'],
    )
    source_config = json.loads((upstream / 'config.json').read_text())
    verify_source(source_config)
    source_template = upstream / 'chat_template.jinja'
    if not source_template.is_file():
        raise RuntimeError('Missing upstream chat_template.jinja')

    command = [sys.executable, '-m', 'mlx_lm.convert', '--hf-path', str(upstream), '--mlx-path', str(output)]
    if args.variant == '8bit':
        command += ['-q', '--q-bits', '8', '--q-group-size', '64', '--q-mode', 'affine']
    else:
        command += ['--dtype', 'bfloat16']
    print('[CONVERT]', ' '.join(command), flush=True)
    subprocess.run(command, check=True)

    target_config_path = output / 'config.json'
    target_config = json.loads(target_config_path.read_text())
    target_config = patch_for_mlx(source_config, target_config)
    if args.variant == '8bit':
        expected_quantization = {'mode': 'affine', 'bits': 8, 'group_size': 64}
        if target_config.get('quantization') != expected_quantization:
            raise RuntimeError(f'Unexpected quantization config: {target_config.get("quantization")!r}')
    target_config_path.write_text(json.dumps(target_config, indent=2) + '\n')
    shutil.copy2(source_template, output / 'chat_template.jinja')
    if sha256(source_template) != sha256(output / 'chat_template.jinja'):
        raise RuntimeError('Chat template mismatch')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if not (output / name).is_file():
            raise RuntimeError(f'MLX conversion did not write {name}')
    output.joinpath('README.md').write_text(card(args.variant, revision))

    shards = sorted(output.glob('model*.safetensors'))
    if not shards:
        raise RuntimeError('No safetensors written')
    receipt = {
        'source_repo': SOURCE,
        'source_revision': revision,
        'mlx_lm_version': version('mlx-lm'),
        'mlx_version': version('mlx'),
        'transformers_version': version('transformers'),
        'variant': args.variant,
        'quantization': {'mode': 'affine', 'bits': 8, 'group_size': 64} if args.variant == '8bit' else None,
        'source_config_sha256': sha256(upstream / 'config.json'),
        'chat_template_sha256': sha256(output / 'chat_template.jinja'),
        'mlx_rope_theta': THETA,
        'weights': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in shards},
    }
    (output / 'CONVERSION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'[DONE] {output}', flush=True)
    print('[NEXT] python smoke.py --model', output, flush=True)


if __name__ == '__main__':
    main()
