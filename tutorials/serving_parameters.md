# Serving parameters, key by key

Every optimized config in `thetalker/deploy/` is a handful of documented keys on
top of the file vllm-omni ships. This is what each of them does and where it
lives.

A vllm-omni TTS pipeline runs in stages. **Stage 0** is the language model that
produces codec codes; **stage 1** is the audio decoder that turns codes into
PCM. Some pipelines are single-stage. Stage settings live under `stages:`, keyed
by `stage_id`; the streaming settings live in the `connectors:` block, which is
shared by the pipeline.

## Standard vLLM keys, per stage

| Key | What it does |
|---|---|
| `max_num_seqs` | how many requests the stage admits at once. The ceiling on concurrency for the whole pipeline is the smallest `max_num_seqs` on the path. On an audio decoder this is also the number of per-request decoder states it can hold, so a value of 1 serialises every decode. |
| `gpu_memory_utilization` | the fraction of **total** device memory the stage may use. It is applied per stage, so on a two-stage pipeline the fractions add up: 0.85 plus 0.12 does not fit one card next to a second CUDA context and its graph pools. |
| `max_num_batched_tokens` | the per-step token budget for chunked prefill. It is not a per-request length limit (`max_model_len` is), but warmup allocates one dummy batch of the full budget, and an eager attention implementation allocates a score tensor that grows with the square of it. |
| `dtype` | the dtype the whole pipeline is constructed and its weights loaded under. Changing it from fp32 to bf16 halves the weight footprint and moves the matmuls onto tensor cores. |
| `enforce_eager` | disables CUDA graph capture for the stage. |

## vllm-omni streaming keys, in `connectors:`

| Key | What it does |
|---|---|
| `initial_codec_chunk_frames` | the size of the **first** audio chunk the decoder emits, in codec frames. This is the player's whole buffer until the second chunk arrives. |
| `codec_chunk_ramp` | a list of chunk sizes for the first chunks, in codec frames, the last entry equal to `codec_chunk_frames`. When set it replaces `initial_codec_chunk_frames`. A ramp of `[2, 4, 8, 16, 32, 75]` lets the first byte leave after 2 frames and produces each next chunk while the previous one plays. Every chunk is one decoder call, so a ramp costs throughput. |
| `codec_chunk_frames` | the size of **every later** chunk, in codec frames. Larger chunks mean fewer, bigger decode calls: less per-call overhead, more audio per delivery, and a longer wait between deliveries. |
| `codec_left_context_frames` | how many previous frames the decoder sees while decoding a chunk. |
| `ref_code_context_frames` | how many frames of the reference clip's codes the decoder keeps as context on a voice-cloning request. |
| `decode_cudagraph_batch_sizes` | the request-batch sizes the decoder captures graphs for. It ships commented out, and the fallback is a single bucket of 1, so by default the decoder replays one graph per request per chunk and never batches across requests. See below. |
| `decode_batch_max_size` | how many requests the codec decodes in one batch. The code default is 0, meaning unlimited, so omitting the key from a connector block is not the same as leaving it alone. |
| `unified_decode_graph_max_batch_size` | the largest batch that still runs under a captured graph. It must move together with `max_num_seqs`: raising admission alone leaves every batch above this limit off the graph, so the extra admission buys nothing. |

A codec frame is a fixed slice of audio: 80 ms on Qwen3-TTS, whose codec runs at
12.5 Hz. So a 1-frame first chunk is 80 ms of audio and a 16-frame first chunk
is 1.3 s.

**The `connectors:` block is replaced, not merged.** vllm-omni's `base_config`
merge deep-merges `stages:` by `stage_id`, but replaces every other top-level key
wholesale. An overlay that changes one connector key must therefore restate the
whole block: a key left out reverts to its class default, not to the base's
value. This is why the Qwen3-TTS overlays in `thetalker/deploy/` repeat
settings they do not change, and why `decode_batch_max_size` appears in them at
the base's own value.

## The optimized configuration, per model

| Model | `max_num_seqs` stage 0 / 1 | First / regular chunk, frames | Other |
|---|---|---|---|
| Qwen3-TTS 1.7B and 0.6B | 64 / 64, unchanged | 16 / 75 | `ref_code_context_frames: 72`, `decode_batch_max_size: 4`, `decode_cudagraph_batch_sizes: [1, 2, 4, 8]` |
| VoxCPM2 | 32 | shipped values | `unified_decode_graph_max_batch_size: 32` |
| MOSS-TTS 8B | 32 / 8 | not streamed | GPU memory utilization 0.30 on both stages, `max_num_batched_tokens: 65536` on stage 1 |
| OmniVoice | shipped value | not streamed | eager; `dtype` fp32 to bf16. The shipped `dtype: float32` cannot be raised to bf16 by configuration alone on 0.28.0; see [known_issues_0.28.md](known_issues_0.28.md) |

Read the exact delta of any row in the file itself: each `*_optimized.yaml` is an
overlay on its sibling `*_default.yaml` and contains nothing but the changed
keys and a header explaining them.

## Deriving `decode_cudagraph_batch_sizes`

The list is the set of request-batch sizes the codec decoder captures a CUDA
graph for. vllm-omni 0.28.0 ships the key commented out in every Qwen3-TTS
config, and the fallback is a single bucket of 1. A server left that way replays
one graph per request per chunk: eight concurrent streams cost eight replays of
a batch-1 graph rather than one replay of a batch-8 graph, and the decoder never
batches across requests at all.

Setting `[1, 2, 4, 8]` covers the load the configuration is tuned for. Any batch
larger than the top bucket falls out of the graph and decodes eagerly, and any
batch between two buckets is padded up to the next one, so the list should end
at or above the concurrency the server is expected to carry.

The cost is startup time and memory. Each bucket multiplies the number of graphs
captured at startup, and on an H100 80GB the Qwen3-TTS 1.7B footprint after
startup grows from 28.5 GiB with the single default bucket to 41.5 GiB with four
buckets. Adding a chunk ramp on top multiplies it again, because every ramp entry
is its own decode window.

There is also a `decode_cudagraph_capture_sizes` key, which named decode window
lengths rather than batch sizes. **On vllm-omni 0.28.0 it is read and ignored
under `async_chunk`**: the decoder derives its capture shapes from the chunk
settings itself. Do not attribute any part of a measured gain to it on 0.28.0.

## What is not a serving parameter

Two things in the published tables are not reachable by editing a config, and are
labelled as such: OmniVoice's bf16 path and MOSS-TTS starting at all on stock
0.28.0. Both are code, not configuration. See
[known_issues_0.28.md](known_issues_0.28.md).
