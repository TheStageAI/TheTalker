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
- **Software.** vLLM 0.30.0 (cu129 wheel), vllm-omni 0.30.0, torch 2.13, CUDA
  12.9, driver 580.
- **Load.** Closed loop: `cN` means N requests in flight, a new one issued as
  each finishes. The published curves run from c1 to c64.
- **Voice.** Qwen3-TTS CustomVoice uses the preset speaker Ryan. The cloning
  models use one reference clip, shipped as `benchmark/reference/ref.wav`
  (4.64 s, 24 kHz mono, synthesized; see its README). OmniVoice and MOSS-TTS
  call a registered voice: the clip is uploaded once through
  `POST /v1/audio/voices` after the server is up, and every request names that
  voice (`"voice": <name>`) instead of carrying the clip. This measures the
  steady-state speed of a registered voice, which is how public TTS benchmarks
  call a hosted voice. VoxCPM2 receives the clip and its transcript with every
  request (`--ref-audio`), and so does every word error rate run. The transcript
  is `benchmark/reference/ref.txt`, passed with `--ref-text` (or `REF_TEXT`);
  the speed points were measured with a transcript one word shorter, which
  does not change speed.
- **MPS.** Qwen3-TTS and MOSS-TTS run under an NVIDIA MPS daemon on both servers
  (`MPS=1 examples/serve.sh`; thestage-vllm-omni starts it by itself).
- **Size of a point.** Scored requests are 100 at 1 stream, 100 at 2, 200 at 4,
  and 300 from 8 streams through 64, after a warmup. The warmup requests
  (`--warmup`, on top of `--requests`) run as their own phase at the point's
  concurrency, and every one of them finishes before the first scored request is
  sent. They are excluded from the reported statistics, and `aggregate_RTFx`
  divides the scored audio by the wall-clock span of the scored requests only.
  Draining matters: a warmup request still in flight inside the scored window
  would use GPU time whose audio is not counted, and read RTFx low. The warmup is per point, not one number for the sweep:
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
    --register-voice benchmark/reference/ref.wav \
    --ref-text "$(cat benchmark/reference/ref.txt)" \
    --concurrency 8 --requests 300 --warmup 8 --seed 42 \
    --out-dir out/c8
```

`--register-voice` uploads the clip once, as voice `thetalker_ref` (`--voice`
names it), before the first request; later points on the same server pass
`--voice thetalker_ref` alone. `thetalker-voice --name <name> --ref-audio <wav>
--ref-text "<transcript>"` registers a voice without running a point.

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
| `--concurrency` / `--requests` | closed-loop load: N scored requests, `--concurrency` in flight at a time |
| `--arrival-rate` | Poisson req/s instead of closed loop (0 = one wave of `--concurrency`) |
| `--warmup` | N warmup requests, run as their own phase and drained before the `--requests` scored ones start; excluded from the stats and from the RTFx window |
| `--seed` | seed of the Poisson schedule |
| `--max-new-tokens` | server-side cap on generated codec tokens (default 256, the protocol value) |
| `--max-error-rate` | share of scored requests that may error before the run is called a failure (default 0.05) |
| `--continuity-threshold-s` | underrun budget in seconds (default 0.1 s, matching vllm-omni) |
| `--task-type` | `Base` (voice clone: a registered voice, or `--ref-audio`/`--ref-text` inline), `CustomVoice` (`--speaker`), `VoiceDesign` (`--instruct`) |
| `--register-voice` / `--voice` | Base task: upload a wav once as voice `--voice` (default `thetalker_ref`) and call it by name; `--voice` alone calls a voice registered earlier |
| `--stream-format` | `audio` (raw PCM chunks, default) or `sse` (OpenAI `speech.audio.delta` events) |
| `--save-audio-limit` | how many decoded wavs to keep; raise it to feed the WER pipeline |

`python benchmark/run_benchmark.py --help` lists the rest.

**Exit status.** 0 when the point is usable. 1, with one line on stderr, when no
request completed or more than `--max-error-rate` of the scored requests errored.
A run against a server that is not up yet still writes a well-formed
`summary.json` of zeros, so the exit code is what tells a script the difference
between that and a measurement.

# Reproducing each table

One model, both columns, both concurrencies. The Qwen3-TTS rows are
CustomVoice with the preset speaker Ryan, so they need no reference clip:

```shell
# 1: vllm-omni column: stock vllm-omni with this repository's deploy file
MPS=1 examples/serve.sh qwen3tts17b 8091 &
TASK_TYPE=CustomVoice SPEAKER=Ryan examples/run_sweep.sh out/qwen3tts17b_vllm_omni 127.0.0.1:8091
# stop the server, verify the GPU is empty

# 2: thestage-vllm-omni column: the library installed, paid session, no deploy file
vllm-omni serve "$(thetalker-deploy --model-id qwen3tts17b)" --omni --trust-remote-code --port 8091 &
TASK_TYPE=CustomVoice SPEAKER=Ryan examples/run_sweep.sh out/qwen3tts17b_thestage 127.0.0.1:8091
```

Qwen3-TTS 0.6B is the same with `qwen3tts06b`.

The cloning models (VoxCPM2 2B, OmniVoice 0.6B, MOSS-TTS 8B) use the shipped
reference clip, `benchmark/reference/ref.wav`, with its transcript
`benchmark/reference/ref.txt`. A different clip changes the numbers; `examples/get_reference.py` is for the examples only.
`run_sweep.sh` registers `REF_AUDIO` once, on its first point, and every request
calls that voice by name; `INLINE_REF=1` sends the clip and transcript with
every request instead. VoxCPM2 2B streams at 48 kHz, so its
sweep needs `SAMPLE_RATE`, or every audio duration and RTFx is off by 2x:

```shell
export REF_AUDIO=benchmark/reference/ref.wav
export REF_TEXT="$(cat benchmark/reference/ref.txt)"

# vllm-omni column; the thestage-vllm-omni column is the same sweep against
# `vllm-omni serve <model> --omni --trust-remote-code` with the library installed
examples/serve.sh voxcpm2 8091 &
INLINE_REF=1 SAMPLE_RATE="$(thetalker-deploy --sample-rate voxcpm2)" examples/run_sweep.sh out/voxcpm2_vllm_omni 127.0.0.1:8091

examples/serve.sh omnivoice 8091 &
examples/run_sweep.sh out/omnivoice_vllm_omni 127.0.0.1:8091

MPS=1 examples/serve.sh moss 8091 &
examples/run_sweep.sh out/moss_vllm_omni 127.0.0.1:8091
```

Stop each server and verify the GPU is empty before the next one.

The published vllm-omni column is the deploy file at 32 streams, on CustomVoice
with speaker Ryan for Qwen3-TTS. `run_sweep.sh` also writes c8. Read
`out/<config>/c32/summary.json`:

| table | key |
|---|---|
| Throughput | `aggregate_RTFx` at c32 |
| Stream continuity | `continuity_ok_pct` |
| First audio under load | `ttfa_p50` |
| Cost | `aggregate_RTFx` at c32, with the input characters of the requests delivered without a stall |

The thestage-vllm-omni column (command 2) needs that library installed and a
TheStage token; with no
`--deploy-config` the library applies its own overlay, so that column is not
reproducible from this repository alone. `examples/serve.sh` sets
`THESTAGE_VLLM_OMNI=0`, so command 1 serves stock vllm-omni even with the library
installed. For Qwen3-TTS and MOSS-TTS 8B, command 1 is measured under an NVIDIA
MPS daemon: `MPS=1 examples/serve.sh <name>` (README, Quick start step 4).

# Quality: word error rate

WER is a check that a serving parameter did not change what the model says, not
a model-quality benchmark. It is not computed by this client. The pipeline:

1. Run one point at c8 over `prompts/seed_tts_eval_en_200.jsonl` with
   `--save-audio-limit 200`, so every scored request leaves a wav, on the Base
   checkpoint, with the reference clip and its exact transcript sent with every
   request: `--ref-audio` with `benchmark/reference/ref.wav`, `--ref-text` with
   the contents of `benchmark/reference/ref.txt`.
2. Transcribe each wav with `openai/whisper-large-v3`, greedy, English.
3. Normalise both the prompt text and the transcript with the standard Whisper
   normaliser (lowercase, punctuation removed), then score word error rate per
   utterance against the prompt text in `quality_manifest.jsonl`.
4. Report the pooled rate, total word edits divided by total reference words,
   with a 95% interval bootstrapped over sentences (the published table prints
   half of it after ±). A serving-parameter change is accepted when the two
   configurations are statistically indistinguishable; a single utterance above
   50% is a failure to investigate, not noise.

WER measures intelligibility, not voice similarity. Speaker similarity on the
cloning path is not measured here.

# GPU memory

The client never touches the GPU, so memory is read from outside. Run
`benchmark/gpu_monitor.py` next to the server for the duration of a point:

```shell
python benchmark/gpu_monitor.py --duration 600 --out out/qwen3tts17b_vllm_omni/gpu.json
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
second one on inputs built to break them. `test_deploy_cli.py` covers `thetalker-deploy` name resolution
and checks that every shipped config is self-contained. `test_deploy_configs.py`
loads every shipped config through vllm-omni's own loader and, on vllm-omni
0.30.0, checks that the keys each one changes against that release's deploy file
are the ones its header lists; it skips cleanly when `vllm_omni` is not
installed, and its equivalence check against the originally measured files
needs `TTS_BENCH_MEASURED_DEPLOY_DIR` (those files are not redistributed).

# Prompt sets

`prompts/seed_tts_eval_en_100.jsonl` and `prompts/seed_tts_eval_en_200.jsonl`
are 100 and 200 English utterances from
[Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval), one JSON
object per line with at least a `text` field. These files keep Seed-TTS-Eval's
own upstream license and terms, not this repository's MIT license.
