# Code-localization diagnostic (work in progress)

This is a small **15-task diagnostic**, not a claim of upstream accuracy parity.
It uses `psf/requests` at commit `611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60`.
Each question has hand-labelled file and 1-based inclusive line-range anchors
that are checked against the pinned source checkout before any model runs.
The tasks and scoring rule are fixed *before* collecting MLX-8bit/BF16 outcomes.

We use LightOn's **existing** `colgrep --agent` implementation for the search
loop, sandboxed read-only `terminal` tool, search index, observations, and the
`finish` submission protocol. A locally running MLX-LM server supplies only
text completions through its `/v1/completions` endpoint. This integration is
**not yet validated end to end**; special-token/streaming incompatibility is
possible even if standalone `smoke.py` succeeds.

## macOS setup

From the repository root (Python virtual environment from README activated):

```sh
brew install lightonai/tap/colgrep
python -m unittest discover -s tests -v
python -m eval.benchmark prepare
```

`prepare` clones the public repo into the ignored `work/eval/corpus/requests`,
checks its exact Git revision and clean state, verifies all golden text anchors,
and builds the ColGrep semantic search index. XDG config and data directories
are isolated under ignored `work/eval/` and do **not** change personal ColGrep
settings. Re-run `prepare` to verify the checkout; it will not reset changes.

If the checkpoint is still under your previous starter directory, reuse it
without copying 2.67 GB (adjust the source path if necessary):

```sh
mkdir -p release
ln -s ~/Downloads/colgrep-agent-mlx-starter/release/ColGrep-Agent-2B-MLX-8bit \
  release/ColGrep-Agent-2B-MLX-8bit
```

Alternatively download the published checkpoint into this clone using:

```sh
hf download imaadd05/ColGrep-Agent-2B-MLX-8bit \
  --local-dir release/ColGrep-Agent-2B-MLX-8bit
```

Start the local server in **terminal A** (MLX 8-bit already converted):

```sh
python -m mlx_lm server \
  --model "$(pwd)/release/ColGrep-Agent-2B-MLX-8bit" \
  --host 127.0.0.1 --port 8080
```

In **terminal B**, first test two tasks, then run all 15:

```sh
python -m eval.benchmark run --name mlx8-probe --limit 2 \
  --model release/ColGrep-Agent-2B-MLX-8bit
python -m eval.score results/eval/mlx8-probe.json

python -m eval.benchmark run --name mlx8-full \
  --model release/ColGrep-Agent-2B-MLX-8bit
python -m eval.score results/eval/mlx8-full.json \
  --output results/eval/mlx8-full.score.json
```

The runner checks that `/v1/models` advertises the **exact local checkpoint**,
configures ColGrep's prompt/tools/template/seed/sampling and 10-turn limit,
uses `--no-update` to reuse the prepared search index, and records each full
ColGrep JSON outcome, time, and trace checksum. `--name` is immutable: repeat
runs must use different names; partial results are saved after every task.

If the probe ends with `MalformedRetries` or `LlmError`, inspect
`results/eval/traces/mlx8-probe/*.json` and the saved `stderr_tail` before
claiming anything about model quality. In particular, verify that MLX-LM's
`/v1/completions` preserves literal `<function>` and `<param>` markup in
streaming chunks and that the final-turn `finish` prefill works. The server's
support for `/v1/completions` alone does not prove behavioral compatibility.

## BF16 comparison (optional)

Convert the same pinned source as a BF16 MLX model:

```sh
python convert.py --variant bf16
```

Stop the 8-bit server, start a new server for
`release/ColGrep-Agent-2B-MLX-bf16` on the **same port**, then run:

```sh
python -m eval.benchmark run --name bf16-full \
  --model release/ColGrep-Agent-2B-MLX-bf16
python -m eval.score results/eval/mlx8-full.json \
  --compare results/eval/bf16-full.json
```

Do not run both large MLX models simultaneously on a 16 GB Mac. Both runs
share a pinned source checkout and indexed code, but maintain separate agent
settings. Repeated seeds/trials and larger repositories would be necessary to
make quantitative equivalence claims.

## Scoring contract

- **Only submitted `finish` locations count.** When `submitted=false`, the
  official CLI can return inspected locations as a fallback; these score 0.
- File metric: per-task macro file precision/recall/F1, matching paths exactly.
- Range metric: one-to-one matching of predicted and gold ranges on the same
  file at **inclusive interval IoU >= 0.25**. A broad whole-file range is
  penalized. Location precision/recall/F1 are macro-averaged over tasks.
- We also report mean best gold interval IoU, submitted fraction, invalid
  location count, error task count, median elapsed seconds and total tokens.
- A malformed, out-of-repository or unbounded submitted location fails closed:
  **no partial credit** for that task.
- Scoring checks the byte hash of the manifest, pinned commit, and exact task
  identities. Partial run outputs explicitly report `complete_manifest=false`.

## Limitations

The 15 tasks are developer-authored, public and cover **one** repository.
They are for catching obvious quantization or integration regressions, not
independent statistical validation. Results depend on the ColGrep version,
index state, sampling and the server's tokenization/streaming. Because
completion timing includes tool activity and any server overhead, report
latency separately from token throughput. Do not characterize parity or
improvement from small count differences or this diagnostic alone.
