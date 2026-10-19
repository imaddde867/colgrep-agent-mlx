# Validation

Validation record for the 8-bit MLX conversion. Last updated 2026-10-19.

## Artifacts

| Item | Value |
|---|---|
| Source | `lightonai/colgrep-agent-2B` |
| Source revision | `f798a61129c6decb5166864e0a3c98373e62c764` |
| MLX release | https://huggingface.co/imaadd05/ColGrep-Agent-2B-MLX-8bit |
| Published commit | `23b8baac392f726b3ac8d9b14b6419f745953420` |
| Quantization | affine, 8-bit, group size 64 |
| Weight file | `model.safetensors`, 2,674,327,346 bytes |
| Weight SHA-256 | `d24593d6d59596260e9df3383525f5726930651789a0204094c82e5e42fc8eb2` |
| Chat template SHA-256 | `cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c` |

The published checkpoint was not modified during repository preparation. Its local weight hash matches `CONVERSION_RECEIPT.json` in the model directory.

## Environment

| Item | Value |
|---|---|
| Hardware | Apple Silicon M4, 16 GB unified memory |
| OS | macOS 27.0.1, arm64 |
| Python | 3.12.15 |
| MLX | 0.32.4 |
| MLX-LM | 0.32.0 |
| Transformers | 5.20.1 |
| huggingface_hub | 1.33.0 |

## RoPE correction

The source config stores `rope_parameters.rope_theta = 5000000`. MLX-LM 0.32.0 reads a top-level `rope_theta` for Llama models and otherwise uses `10000`. The conversion writes the top-level compatibility key while leaving the model weights and original `rope_parameters` intact.

Covered by tests:

- The source theta must equal `5000000`.
- An existing conflicting top-level theta is rejected.
- The patched config retains both representations.
- The loaded model reports the corrected theta during the smoke test.

## Tests executed

```bash
python -m unittest discover -s tests -v
python -m compileall -q convert.py smoke.py tests
```

Results:

| Check | Result |
|---|---|
| Conversion/config and contract tests | 7 passed |
| Tool-call XML parser tests | 13 passed |
| Total unit tests | 20 passed |
| `compileall` | passed |
| Trained system prompt fingerprint | matched |
| Tool schema fingerprint | matched |

The parser tests cover valid calls, CDATA arguments, complete `<function>` elements, truncated calls, unknown tools, unknown arguments, missing required arguments, malformed XML, unterminated CDATA, stray closing tags, and unusable `colgrep` calls.

## Smoke test executed

```bash
python smoke.py --model release/ColGrep-Agent-2B-MLX-8bit
```

Both prompts passed. For each prompt the model loaded with RoPE theta `5000000`, the original tool schema rendered with thinking disabled, and the first action was a complete `<function name="colgrep">` element containing a nonempty `query` argument. Both calls parsed as XML and passed the structural and schema checks.

The smoke test only validates load configuration and the first generated action. It does not execute generated commands, run a search loop, or measure retrieval accuracy.

## Not validated

- Retrieval quality or equivalence with the upstream BF16 checkpoint.
- Multi-turn ColGrep agent trajectories or `finish` accuracy.
- Compatibility with the existing ColGrep CLI, which runs GGUF through llama.cpp.
- Throughput, latency, or memory benchmarks against other runtimes.
- Behavior on Apple Silicon configurations other than the one listed above.
