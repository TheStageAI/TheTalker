# Known issues on vllm-omni 0.28.0

Three of the five models in the Benchmarks tables run into a defect of the stock
0.28.0 release. Each one is the reason a table cell reads the way it does, so
they are recorded here rather than left as a footnote.

## MOSS-TTS 8B does not start with the public codec checkpoint

Stock vllm-omni 0.28.0 refuses to start MOSS-TTS. Codec version detection keys on
a config field that the public codec checkpoint does not carry, so the pipeline
cannot decide which codec it is loading and construction fails.

**Effect on the tables.** Both MOSS-TTS runs are on a tree with that detection
restored. That is why both ratio cells of the model's Throughput row read n/a:
the two steps cannot be separated, because the default file only starts with the
library's codec fix, so there is no stock-0.28.0 baseline to divide by. Measured
together against the vllm-omni default, with both sides carrying the codec fix,
they are 1.55x / 3.30x.

**Effect on you.** `thetalker-deploy moss_default` and `moss_optimized` are the
configurations that were measured, and they are correct, but neither will start a
server on stock 0.28.0. The fix ships inside thestage-vllm-omni, not in a deploy file.

## OmniVoice cannot reach bf16 by configuration

The OmniVoice deploy file vllm-omni ships sets `dtype: float32`, and the model
runs in fp32. Setting `dtype: bfloat16` is not enough on stock 0.28.0: the bf16
path builds an attention mask in fp32 and passes it to a bf16 query, and the
forward pass fails on the dtype mismatch.

**Effect on the tables.** Both ratio cells of OmniVoice's Throughput row read
n/a as well: its only change, bf16, needs the library, so the two steps cannot be
separated either. Measured together against the fp32 default they are 6.07x /
6.45x, of which bf16 alone is 4.04x and the rest is request batching.
`omnivoice_optimized` in this repository carries the one changed key, `dtype:
bfloat16`, because that is the configuration that was measured; it needs the mask
fix underneath it to run.

**If you only want the memory saving**, note that fp32 weights are twice the
footprint for the same output; the row in the Support Matrix reports the bf16
figure.

## VoxCPM2 lost its memory tiers

vllm-omni 0.26.0 selected VoxCPM2's admission cap from the device's memory
through a `device_memory_tiers` ladder, which resolved to 32 on an 80GB card.
Release 0.28.0 dropped the tiers and ships the small-card floor of 8 on every
device, so the same file that reached 32 on an H100 a release earlier now admits
8.

**Effect on the tables.** This is why VoxCPM2's optimized parameters are worth
1.45x at 32 concurrent streams on 0.28.0 while being worth nothing a release
earlier: the two keys are restoring a setting the framework used to derive.

**Effect on you.** `voxcpm2_optimized` pins `max_num_seqs: 32` and
`unified_decode_graph_max_batch_size: 32` unconditionally, which is the H100 80GB
point. On a smaller card serve `voxcpm2_default` instead: the pinned pair will
try to admit 32 requests regardless of how much memory the device has. The two
keys have to move together -- raising admission alone leaves every batch above
the graph limit off the unified decode graph, so the extra admission buys
nothing.

## General

- **`connectors:` is replaced, not merged.** An overlay that changes one
  streaming key must restate the whole `connectors:` block; a key left out
  reverts to its class default rather than to the base's value. This has silently
  changed `decode_batch_max_size` from the base's 4 to the code default of
  unlimited in configs that forgot it.
- **`decode_cudagraph_capture_sizes` is ignored under `async_chunk` on 0.28.0.**
  The decoder derives its capture shapes from the chunk settings. The key is
  still present in the shipped configs so they read as the configurations they
  were measured as; do not attribute a gain to it on this release.
- **`platforms:` blocks cannot be removed by an overlay.** The loader always
  unions a base's platform entries into the merged result. Two of the configs
  here therefore inherit a non-CUDA platform block from their base. It is inert
  on the CUDA path these numbers were taken on.
