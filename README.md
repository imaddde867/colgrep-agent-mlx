# ColGrep-Agent-2B MLX

Reproducible 8-bit [MLX](https://github.com/ml-explore/mlx-lm) conversion of LightOn AI's **ColGrep-Agent-2B** for Apple Silicon.

ColGrep-Agent-2B is a specialized repository code-localization agent built on MiniCPM5-2B. Given a software problem, it searches a repository with `colgrep` and `terminal`, verifies the code, and submits file/line ranges with `finish`. It is **not** a general coding assistant.

- **Converted checkpoint:** [imaadd05/ColGrep-Agent-2B-MLX-8bit](https://huggingface.co/imaadd05/ColGrep-Agent-2B-MLX-8bit)
- **Original model:** [lightonai/colgrep-agent-2B](https://huggingface.co/lightonai/colgrep-agent-2B)
- **Original project:** [lightonai/next-plaid](https://github.com/lightonai/next-plaid)

This is an unofficial community conversion, not affiliated with LightOn AI.

## Specifications

| | |
|---|---|
| Source checkpoint | `lightonai/colgrep-agent-2B` |
| Source revision | `f798a61129c6decb5166864e0a3c98373e62c764` |
| Architecture | Llama (`LlamaForCausalLM`) / MiniCPM5-2B, 2.52B parameters |
| Quantization | 8-bit affine, group size 64 |
| Runtime | MLX-LM 0.32.0, MLX 0.32.4, Transformers 5.20.1 |
| RoPE theta | `5000000` |
| Context length | 131,072 tokens |
| Weights | 2.67 GB `model.safetensors` |
| Hardware | Apple Silicon; validated on an M4 with 16 GB memory |
| Model license | Apache-2.0 |

### RoPE compatibility fix

The source config stores `rope_parameters.rope_theta = 5000000`, but MLX-LM 0.32.0's Llama implementation reads a top-level `rope_theta` and otherwise silently falls back to `10000`. The conversion adds the top-level key while preserving the original configuration. No model weights are altered. The behavior and mismatch checks are covered by tests.

## Quickstart

Requires macOS on Apple Silicon, Python 3.12+, about 10–15 GB of free disk space, and internet access to Hugging Face.

```bash
git clone https://github.com/imaddde867/colgrep-agent-mlx.git
cd colgrep-agent-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Convert locally

```bash
python convert.py
```

This downloads the pinned source revision into `work/source/` and writes the 8-bit checkpoint to `release/ColGrep-Agent-2B-MLX-8bit/`. The script refuses to overwrite an existing output directory. Use `--revision <sha>` to convert a different upstream commit, or `--variant bf16` for an unquantized MLX reference.

### Smoke test

```bash
python smoke.py --model release/ColGrep-Agent-2B-MLX-8bit
```

The smoke test verifies model loading, the loaded RoPE theta, rendering of the original chat template with thinking disabled, and generation of two first tool calls. Generated calls are parsed as XML and checked for complete `<function>` elements, valid CDATA, recognized tool names, schema-valid arguments, and a usable `colgrep` query, pattern, or command. It does **not** execute model-generated commands. Outputs are written to the git-ignored `results/` directory.

## Reproducibility

- The default source revision is pinned in `convert.py` and can be overridden with `--revision`.
- Conversion refuses to run if the output directory already exists.
- The upstream config is validated before conversion.
- Quantization settings are explicit: affine, 8-bit, group size 64.
- The upstream chat template is copied byte-for-byte and verified by SHA-256.
- The source config must contain the expected architecture, EOS IDs, and RoPE theta.
- `release/ColGrep-Agent-2B-MLX-8bit/CONVERSION_RECEIPT.json` records the source revision, package versions, quantization, config and template hashes, and output weight size/hash.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the validation record.

## Validation status

Completed:

- 20 static/unit tests pass locally.
- Real MLX smoke test passes on Apple Silicon: both prompts produced complete, parseable XML tool calls with nonempty `colgrep` queries.
- The checkpoint was published to Hugging Face at commit `23b8baac392f726b3ac8d9b14b6419f745953420`.

Not tested or demonstrated:

- End-to-end retrieval quality or parity with the upstream BF16 model.
- Multi-turn search trajectories or production agent reliability.
- Integration with the existing ColGrep CLI, which currently runs GGUF through llama.cpp.
- Performance or memory benchmarks against other runtimes.

The smoke test establishes that the model loads and emits a structurally valid first action. It is not an agent-quality evaluation.

## Limitations

- This MLX checkpoint is **not** a drop-in replacement for `colgrep --agent`. The official CLI currently uses GGUF/llama.cpp; an MLX-backed harness is separate integration work.
- 8-bit quantization may change model behavior relative to the source checkpoint. No quality-equivalence claim is made.
- Only the initial tool call is checked. Whether the model finds correct files and line ranges has not been evaluated here.
- Validated on one Apple Silicon machine; other Macs and memory configurations are untested.

## Credits and licensing

- **LightOn AI** for [ColGrep-Agent-2B](https://huggingface.co/lightonai/colgrep-agent-2B) and the [next-plaid](https://github.com/lightonai/next-plaid) harness.
- The **MiniCPM5** authors for the base model architecture and training work.
- The **MLX** and **MLX-LM** teams for the Apple Silicon inference framework.
- `harness/system_prompt.txt` and `harness/tools.json` are copied from the upstream Apache-2.0 ColGrep project. The original system prompt and tool schemas are preserved unchanged.

The scripts in this repository are licensed under the [MIT License](LICENSE). That license does not apply to the model weights or upstream harness assets, which remain under their original Apache-2.0 terms. A copy of the Apache License is included in [LICENSE-APACHE](LICENSE-APACHE) for the harness assets and converted model.
