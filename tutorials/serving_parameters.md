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
| `codec_chunk_frames` | the size of **every later** chunk, in codec frames. Larger chunks mean fewer, bigger decode calls: less per-call overhead, more audio per delivery, and a longer wait between deliveries. |
| `codec_left_context_frames` | how many previous frames the decoder sees while decoding a chunk. |
| `ref_code_context_frames` | how many frames of the reference clip's codes the decoder keeps as context on a voice-cloning request. |
| `decode_cudagraph_capture_sizes` | the decode window lengths captured as CUDA graphs. It has to be re-derived whenever the chunk sizes change; see below. |
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
| Qwen3-TTS 1.7B and 0.6B | 64 / 64, unchanged | 16 / 75 | `ref_code_context_frames: 72`, `decode_batch_max_size: 4`, capture sizes retuned to the resulting decode windows |
| VoxCPM2 | 32 | shipped values | `unified_decode_graph_max_batch_size: 32` |
| MOSS-TTS 8B | 32 / 8 | not streamed | GPU memory utilization 0.30 on both stages, `max_num_batched_tokens: 65536` on stage 1 |
| OmniVoice | shipped value | not streamed | eager; `dtype` fp32 to bf16. The shipped `dtype: float32` cannot be raised to bf16 by configuration alone on 0.28.0; see [known_issues_0.28.md](known_issues_0.28.md) |

Read the exact delta of any row in the file itself: each `*_optimized.yaml` is an
overlay on its sibling `*_default.yaml` and contains nothing but the changed
keys and a header explaining them.

## Deriving `decode_cudagraph_capture_sizes`

The capture list is the set of decode window lengths the codec will actually be
asked for. A window it does not cover is padded up to the next captured size, or
falls out of the graph entirely, so the list has to be re-derived whenever a
chunk size moves.

For Qwen3-TTS the windows are:

- **First chunk:** `ref_tail + initial_codec_chunk_frames`. With the reference
  tail at 58 frames and a 16-frame first chunk, that is `58 + 16 = 74`.
- **Steady state:** `ref_tail + codec_left_context_frames + codec_chunk_frames`.
  With 58, 72 and 75, that is `58 + 72 + 75 = 205`.
- **The internal non-streaming window** stays at 325 regardless of the chunk
  settings, and must stay in the list.
- **In between:** the window grows from the first chunk to the steady state as
  the left context fills up, so the ladder covers that range in steps rather than
  jumping from 74 to 205. The shipped list is
  `[74, 91, 109, 128, 149, 170, 190, 205, 325]`.

The same construction with a 1-frame first chunk and 50-frame chunks gives
`59 = 58 + 1` at the bottom and `180 = 58 + 72 + 50` at the top:
`[59, 84, 109, 134, 159, 180, 325]`.

**On vllm-omni 0.28.0 this key is read and ignored under `async_chunk`**: the
decoder derives its capture shapes from the chunk settings itself. It is kept in
the shipped configs so that a config reads the same as the configuration it was
measured as, and because the derivation still applies to releases that honour
it. Do not
attribute any part of a measured gain to it on 0.28.0.

## What is not a serving parameter

Two things in the published tables are not reachable by editing a config, and are
labelled as such: OmniVoice's bf16 path and MOSS-TTS starting at all on stock
0.28.0. Both are code, not configuration. See
[known_issues_0.28.md](known_issues_0.28.md).
