# Table of Contents

- [Protocol](#protocol)
- [Metrics](#metrics)
- [Install](#install)
- [Run](#run)
- [Reproducing each table](#reproducing-each-table)
- [Quality: word error rate](#quality-word-error-rate)
- [GPU memory](#gpu-memory)
- [Tests](#tests)
- [Prompt sets](#prompt-sets)

# Protocol

Every number in the top-level README's Benchmarks section was taken this way.

- **Hardware.** One NVIDIA H100 80GB SXM, one server at a time, GPU verified
  empty before each run.
- **Software.** vLLM 0.28.0 (cu129 wheel), vllm-omni 0.28.0, torch 2.13, CUDA
  12.9, driver 580.
- **Load.** Closed loop: `cN` means N requests in flight, a new one issued as
  each finishes. Points are c8 and c32.
- **Size of a point.** 300 requests, after a warmup of 8 requests at c8 and 100
  at c32. The warmup requests are issued into the same run and excluded from the
  reported statistics (`--warmup`), so the server is at steady state before the
  first scored request. The warmup is per point, not one number for the sweep:
  the codec captures CUDA graphs on first use, and a c32 point needs enough
  warmup that the scored requests are not the ones paying for that.
  `examples/run_sweep.sh` applies 8 at concurrency 8 or below and 100 above it
  unless `WARMUP` is set.
- **Generation cap.** `--max-new-tokens 256`, the client's default and the value
  the tables were measured with. It is recorded in every `summary.json` as
  `max_new_tokens_cap`.
- **Pairing.** Both sides of a comparison run back to back in one session, in
  one server-restart order, on the same prompt set with the same seed. A ratio
  between two numbers taken in different sessions is not reported.
- **Prompts.** Seed-TTS-Eval EN, 100 utterances for speed, 200 for word error
  rate. Requests cycle the file in order, so both sides of a pair see the same
  text for the same request index.
- **Transport.** Streaming over the OpenAI `POST /v1/audio/speech` API with
  `stream: true`, `stream_format: audio`, `response_format: pcm`. The client
  reads with `read1()`, not `read()`: `read()` blocks until the full buffer is
  available, which bunches arrival timestamps and fakes the chunk timeline. A
  response that opens with a `RIFF` header or an SSE envelope is flagged, because
  byte counts then overstate the audio and every byte-derived metric is wrong.
- **What invalidates a point.** Another process on the GPU, or cuDNN debug
  logging left on in the environment.

# Metrics

All of it is computed in `thetalker/metrics.py`, once per request, from the chunk
timeline the client records. Nothing is computed twice or in the client.

| metric | definition |
|---|---|
| RTFx | audio seconds produced per wall-clock second, aggregated over the point (`aggregate_RTFx`); higher is better |
| TTFA, first byte | request sent to first audio byte received, in seconds; lower is better. Connect cost is measured separately (`t_connect_ms`) and excluded |
| TTFA, audible | first byte plus the leading silence the model itself generates |
| continuity | share of requests whose audio never fell behind real-time playback (`continuity_ok_pct`) |

**Continuity model.** A real-time player is fed from the first chunk with no
pre-roll: the playback clock starts at the first chunk's arrival, and the player
runs dry whenever the audio delivered so far is shorter than the time elapsed
since then. The worst such deficit in a request is `buffer_deficit_s`; the
request counts as continuous when it stays within the budget, default 0.1 s
(`--continuity-threshold-s`, `VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S`).
`compute_continuity_stats` is a verbatim port of vllm-omni's own
`vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats` (PR #3618),
so the number is definitionally identical to their `audio_underrun` metric. Each
record also carries the verdict at 0.05 s and 0.2 s
(`continuity_ok_by_threshold`).

**Audible-onset detector.** Leading silence is the offset to the first 5 ms
window whose RMS reaches 5% of that file's own peak and stays there for 20 ms.
Those three constants (`window_ms=5.0`, `rel_threshold=0.05`, `sustain_ms=20.0`
in `audible_onset_offset_s`) are this repository's definition of onset, not a
standard: an "audible TTFA" published by another stack is not known to share
them. The detector may only ever move TTFA later, never earlier -- an empty
buffer, a buffer that opens with sound, and a buffer that never crosses the
threshold all read 0.0.

**Playback models are not interchangeable.** Continuity starts the clock at the
first byte; audible TTFA describes a listener who hears nothing until the onset.
The 0.1 s test is not invariant to that difference, so a continuity verdict is
only meaningful together with the model it was taken under.

# Install

```shell
pip install -r benchmark/requirements.txt
```

Only `numpy` is needed to run the client. The rest are the package's optional
groups: `thetalker[test]` for `pytest`, `thetalker[examples]` for
`examples/get_reference.py`, `thetalker[load]` for
`examples/load_test_locust.py`.

# Run

```shell
python benchmark/run_benchmark.py \
    --port 8091 \
    --texts-jsonl benchmark/prompts/seed_tts_eval_en_100.jsonl \
    --task-type Base \
    --ref-audio reference.wav --ref-text "<the transcript of that recording>" \
    --concurrency 8 --requests 300 --warmup 8 --seed 42 \
    --out-dir out/c8
```

Installed as a package, the same CLI is `thetalker-bench`.

Writes into `--out-dir`:

| file | content |
|---|---|
| `summary.json` | the point: `aggregate_RTFx`, `ttfa_p50`/`ttfa_p95`, `ttfa_audible_p50`/`p95`, `continuity_ok_pct`, `buffer_deficit_p50`/`p95`, `wall_p50`/`p95`, error count, and the settings the point ran with. `run` labels which client produced it (`thetalker`) and `run_id` joins it to `profile` and `point`, so results from several configurations can be concatenated and still be told apart |
| `records.jsonl` | one row per request, including the full chunk timeline (`chunk_arrival_s`, `chunk_bytes`) |
| `schedule.jsonl` | the planned arrival offsets, so a pair can be replayed on an identical schedule |
| `prompt_rows.jsonl` | the prompts as the client read them |
| `quality_manifest.jsonl` | text and wav path per saved request, the input to the WER pipeline |
| `audio/` | up to `--save-audio-limit` decoded wavs |

Key flags:

| flag | meaning |
|---|---|
| `--concurrency` / `--requests` | closed-loop load: N requests, `--concurrency` in flight at a time |
| `--arrival-rate` | Poisson req/s instead of closed loop (0 = one wave of `--concurrency`) |
| `--warmup` | first N requests excluded from the reported stats |
| `--seed` | seed of the Poisson schedule |
| `--max-new-tokens` | server-side cap on generated codec tokens (default 256, the protocol value) |
| `--max-error-rate` | share of scored requests that may error before the run is called a failure (default 0.05) |
| `--continuity-threshold-s` | underrun budget in seconds (default 0.1 s, matching vllm-omni) |
| `--task-type` | `Base` (voice clone, needs `--ref-audio`/`--ref-text`), `CustomVoice` (`--speaker`), `VoiceDesign` (`--instruct`) |
| `--stream-format` | `audio` (raw PCM chunks, default) or `sse` (OpenAI `speech.audio.delta` events) |
| `--save-audio-limit` | how many decoded wavs to keep; raise it to feed the WER pipeline |

`python benchmark/run_benchmark.py --help` lists the rest.

**Exit status.** 0 when the point is usable. 1, with one line on stderr, when no
request completed or more than `--max-error-rate` of the scored requests errored.
A run against a server that is not up yet still writes a well-formed
`summary.json` of zeros, so the exit code is what tells a script the difference
between that and a measurement.

# Reproducing each table

One model, both configurations, both concurrencies. Qwen3-TTS 1.7B shown; every
other model is the same commands with its own config names from
`thetalker-deploy`.

Both runs need the same reference clip and its transcript, since these
checkpoints are served in the voice-cloning task type. `examples/get_reference.py`
fetches an openly licensed one and prints the transcript to put in `REF_TEXT`
(see the top-level README, Quick start step 3).

```shell
export REF_AUDIO=reference.wav
export REF_TEXT="<the transcript of that recording, word for word>"

# 1: the config vllm-omni ships
examples/serve.sh qwen3tts17b_default 8091 &
examples/run_sweep.sh out/qwen3tts17b_default 127.0.0.1:8091
# stop the server, verify the GPU is empty

# 2: the optimized serving parameters
examples/serve.sh qwen3tts17b_optimized 8091 &
examples/run_sweep.sh out/qwen3tts17b_optimized 127.0.0.1:8091
```

The CustomVoice and VoiceDesign checkpoints take the same deploy configs; name
the checkpoint and the task instead of a reference clip:

```shell
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice examples/serve.sh qwen3tts17b_optimized 8091 &
TASK_TYPE=CustomVoice SPEAKER=vivian examples/run_sweep.sh out/cv17_optimized 127.0.0.1:8091
```

`run_sweep.sh` defaults to the protocol above: 300 requests, seed 42, at c8 and
c32, warmup 8 at c8 and 100 at c32, one `--out-dir` per point. Read the numbers off
`out/<config>/c8/summary.json` and `out/<config>/c32/summary.json`:

| table | key |
|---|---|
| Throughput | `aggregate_RTFx`, optimized over default for the ratio |
| Stream continuity | `continuity_ok_pct` |
| First audio under load | `ttfa_p50` and `ttfa_audible_p50` at c8 |
| Cost | `aggregate_RTFx` at c32, with the input characters of the requests delivered without a gap |

The third configuration of the published tables, thestage-vllm-omni, is the same
two commands with that library installed and its own overlay named on
`--deploy-config`; it is not reproducible from this repository alone.

# Quality: word error rate

WER is a check that a serving parameter did not change what the model says, not
a model-quality benchmark. It is not computed by this client. The pipeline:

1. Run one point at c8 over `prompts/seed_tts_eval_en_200.jsonl` with
   `--save-audio-limit 200`, so every scored request leaves a wav.
2. Transcribe each wav with `openai/whisper-large-v3`, greedy, English.
3. Normalise both the prompt text and the transcript with the standard Whisper
   normaliser (lowercase, punctuation removed), then score word error rate per
   utterance against the prompt text in `quality_manifest.jsonl`.
4. Report the mean, the median and the maximum. A serving-parameter change is
   accepted when the two configurations are statistically indistinguishable; a single
   utterance above 50% is a failure to investigate, not noise.

WER measures intelligibility, not voice similarity. Speaker similarity on the
cloning path is not measured here.

# GPU memory

The client never touches the GPU, so memory is read from outside. Run
`benchmark/gpu_monitor.py` next to the server for the duration of a point:

```shell
python benchmark/gpu_monitor.py --duration 600 --out out/qwen3tts17b_optimized/gpu.json
```

It samples `nvidia-smi` and reports peak memory and mean utilisation. It reads
the whole device, so the number is only attributable to the server when nothing
else is running on that GPU.

# Tests

No server, no GPU:

```shell
python -m pytest benchmark -q
```

`test_metrics.py` covers the continuity math, the percentile convention and the
record schema. `test_text.py` and `test_text_adversarial.py` cover the sentence
splitter behind `examples/run_long_text.py` and that script's joining, the
second one on inputs built to break them. `test_deploy_cli.py` covers `thetalker-deploy` name resolution and
`base_config` rewriting. `test_deploy_configs.py` loads every shipped config
through vllm-omni's own loader and checks that what `thetalker-deploy` prints loads
to the same config as the shipped file; it skips cleanly when `vllm_omni` is not
installed, and its equivalence check against the originally measured standalone
files needs `TTS_BENCH_MEASURED_DEPLOY_DIR` (those files are not redistributed).

# Prompt sets

`prompts/seed_tts_eval_en_100.jsonl` and `prompts/seed_tts_eval_en_200.jsonl`
are 100 and 200 English utterances from
[Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval), one JSON
object per line with at least a `text` field. These files keep Seed-TTS-Eval's
own upstream license and terms, not this repository's MIT license.
