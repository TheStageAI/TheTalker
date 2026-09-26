# Reproducing a benchmark row, end to end

One model, both configurations, both concurrency points. Qwen3-TTS 1.7B-Base
is used throughout; every other model is the same steps with its own config
names.

At the end you have four `summary.json` files and the two ratios that make one
row of the Throughput table.

## 0. What you need

- One NVIDIA H100 80GB, idle. Another process on the card invalidates the point.
- This repository and the pinned framework installed: `pip install .[nvidia]`,
  which brings vLLM 0.28.0 and vllm-omni 0.28.0 (README, Quick start step 1).
- A reference wav and its transcript. The Base task type is voice cloning: every
  request carries a reference recording, and both configurations must use the
  same one. A clean mono WAV of 5 to 15 seconds of a single speaker, plus its
  word-for-word transcript. `examples/get_reference.py` fetches an openly
  licensed one and prints the transcript.

Check the GPU is empty before you start:

```shell
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Put the reference in the environment once; every command below reads it from
there:

```shell
python examples/get_reference.py --out reference.wav
export REF_AUDIO=reference.wav
export REF_TEXT="<the transcript it printed>"
```

Unset any `CUDNN_*` debug logging variables in the shell you serve from. cuDNN
writes a line per kernel-plan build when they are set, which costs measurable
time on convolution-heavy paths and does so unevenly, so a contaminated run
cannot be rescaled, only re-measured.

## 1. Configuration 1: the file vllm-omni ships

```shell
examples/serve.sh qwen3tts17b_default 8091
```

Wait for the server to report it is listening. The first start of a checkpoint
compiles kernels and downloads weights, which takes much longer than later
starts; that time is not part of any measurement.

In a second shell:

```shell
examples/run_sweep.sh out/qwen3tts17b_default 127.0.0.1:8091
```

This runs the protocol: 300 requests, warmup 8, seed 42, at c8 and then c32, one
output directory per point. It writes `out/qwen3tts17b_default/c8/summary.json`
and `.../c32/summary.json`.

Optionally, in a third shell, record the memory the server occupies:

```shell
python benchmark/gpu_monitor.py --duration 900 --out out/qwen3tts17b_default/gpu.json
```

Stop the server when the sweep finishes, and confirm the card is empty again
before the next configuration. Two servers alive at once is the most common way
to get a number that cannot be compared to anything.

## 2. Configuration 2: the optimized serving parameters

Same two commands, `_optimized` instead of `_default`:

```shell
examples/serve.sh qwen3tts17b_optimized 8091
```

```shell
examples/run_sweep.sh out/qwen3tts17b_optimized 127.0.0.1:8091
```

Run this immediately after configuration 1, in the same session, on the same
machine. A ratio between two numbers taken on different days is not a
measurement of the change; it is a measurement of the change plus everything
else that moved.

## 3. Read the numbers

```shell
python - <<'PY'
import json, pathlib
for name in ("qwen3tts17b_default", "qwen3tts17b_optimized"):
    for point in ("c8", "c32"):
        s = json.loads(pathlib.Path(f"out/{name}/{point}/summary.json").read_text())
        print(f"{name:26s} {point:4s} RTFx {s['aggregate_RTFx']:7.2f}  "
              f"TTFA p50 {s['ttfa_p50']:.3f}  audible {s['ttfa_audible_p50']:.3f}  "
              f"continuity {s['continuity_ok_pct']:.0f}%  errors {s['errors']}")
PY
```

- **Throughput table:** `aggregate_RTFx` of configuration 2 divided by
  configuration 1, at c8 and at c32.
- **Continuity table:** `continuity_ok_pct` of each configuration.
- **First-audio table:** `ttfa_p50` and `ttfa_audible_p50` of each
  configuration at c8.

`errors` must be 0. A point with errors is not a slower point, it is a different
point: failed requests produce no audio, so they lower the numerator of RTFx and
leave the denominator alone.

## 4. Sanity checks before believing a ratio

- Both configurations ran the same number of scored requests (`n` in
  `summary.json`) over the same prompt file (`prompt_count`, `texts_jsonl`).
- `warmup_excluded` is 8 at c8 and 100 at c32.
- No request has a `ContainerFormat` error in `records.jsonl`. That error means
  the server answered with a WAV header or an SSE envelope instead of raw PCM,
  in which case the byte counts overstate the audio and every derived metric is
  wrong.
- The difference is larger than the spread between two runs of the same
  configuration. If you have never measured that spread on your machine, run
  configuration 1 twice before trusting a ratio under about 1.05x.

## 5. Word error rate

A serving-parameter change must not change what the model says. Run one extra
point per configuration at c8 over the 200-utterance prompt file, keeping every
wav:

```shell
python benchmark/run_benchmark.py --port 8091 \
    --texts-jsonl benchmark/prompts/seed_tts_eval_en_200.jsonl \
    --task-type Base --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
    --concurrency 8 --requests 200 --warmup 8 --seed 42 \
    --save-audio-limit 200 --out-dir out/qwen3tts17b_optimized/wer_c8
```

Then transcribe `audio/*.wav` with `openai/whisper-large-v3` and score against
the `text` field of `quality_manifest.jsonl` under the standard Whisper
normaliser. See
[benchmark/README.md](../benchmark/README.md#quality-word-error-rate).

## 6. The third configuration

The published tables carry a third configuration, thestage-vllm-omni. It is the
same two commands with that library installed and no `--deploy-config`: the
library applies its own overlay by default, which on Qwen3-TTS runs the code
predictor as one persistent GPU kernel. It is not reproducible from this
repository alone.

## A worked smoke run

Before spending an hour on a real point, run a short one and check that the
whole path works. This is the exact sequence, and the output it produced on one
H100 80GB with vLLM 0.28.0 and vllm-omni 0.28.0.

**These numbers are a 20-request smoke test, not a benchmark result.** A point
that short is dominated by ramp-up and by its own tail: it ran for 4.6 seconds
in total, so its RTFx is far below the 50.6 a full 300-request point reaches with
this configuration at c8. Nothing here should be quoted or compared. What it does prove
is that the server, the deploy config, the reference clip and the client agree.

```shell
# 1. reference clip, once
python examples/get_reference.py --out reference.wav
export REF_AUDIO=reference.wav
export REF_TEXT="<the transcript it printed>"

# 2. serve
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --trust-remote-code --port 8091 \
    --deploy-config "$(thetalker-deploy qwen3tts17b_optimized)"

# 3. one short point: 20 requests at c8, first 4 excluded
thetalker-bench --port 8091 \
    --texts-jsonl benchmark/prompts/seed_tts_eval_en_100.jsonl \
    --task-type Base --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
    --concurrency 8 --requests 20 --warmup 4 --seed 42 \
    --out-dir out/smoke_c8
```

`out/smoke_c8/summary.json`, abridged:

```json
{
  "n": 16,
  "warmup_excluded": 4,
  "completed": 16,
  "errors": 0,
  "elapsed_seconds": 4.613,
  "aggregate_audio_seconds": 42.56,
  "aggregate_RTFx": 9.226,
  "ttfa_p50": 0.352,
  "ttfa_p95": 3.348,
  "ttfa_audible_p50": 0.568,
  "buffer_deficit_p50": 0.0,
  "continuity_ok_pct": 100.0,
  "run": "thetalker",
  "run_id": "thetalker__closed_loop__c8",
  "point": "c8"
}
```

`errors: 0` and `continuity_ok_pct: 100.0` are the two fields worth reading at
this size. `aggregate_RTFx` is not.

The single-request client on the same server:

```shell
python examples/run_streaming.py --port 8091 \
    --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
    --text "The quick brown fox jumps over the lazy dog." --out out.wav
```

```
wrote out.wav (3.60 s of audio in 22 chunks)
TTFA first byte : 0.161 s
TTFA audible    : 0.721 s (leading silence 0.560 s)
longest gap     : 0.182 s
playback        : no stall (worst underrun 0.000 s, budget 0.100 s)
```

The 0.560 s of leading silence is the model's own; it is why this repository
reports audible time to first audio next to the first-byte figure.

## What will not reproduce exactly

Sampled TTS output is not bit-reproducible across process reruns on a bf16 GPU
path, even with a fixed seed: the underlying libraries pick reduction and
attention algorithms by internal heuristic, and a tiny numeric difference flips a
sampling decision, after which the two runs diverge. Gate a change with word
error rate and with latency distributions over hundreds of requests, never with
byte-equal audio.
