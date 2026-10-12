# Known issues on vllm-omni 0.28.0

Three of the five models in the Benchmarks tables ran into a defect of the stock
0.28.0 release. The tables are measured on 0.30.0; these notes record what 0.28.0
did and what still applies.

## MOSS-TTS 8B: codec version detection on 0.28.0

Stock vllm-omni 0.28.0 refuses to start MOSS-TTS. Codec version detection keys on
a config field that the public codec checkpoint does not carry, so the pipeline
cannot decide which codec it is loading and construction fails.

**Effect on the 0.28.0 measurements.** Both MOSS-TTS runs were on a tree with
that detection restored, so on 0.28.0 the deploy-file step and the library step
could not be separated: there was no stock-0.28.0 baseline to divide by. Measured
together against the vllm-omni default, with both sides carrying the codec fix,
they were 1.55x / 3.30x.

**Effect on you.** `thetalker-deploy moss` is correct (it loads to the same
configuration as the library's `tuned` overlay), but it does not start a server
on stock 0.28.0; that fix ships inside thestage-vllm-omni, not in a deploy file.
On 0.30.0 it starts on stock vllm-omni with `--init-timeout 1800
--stage-init-timeout 1800` (README, Quick start step 4): the first start's
stage-1 compile outlasts the default 600 s startup timeout.

## OmniVoice cannot reach bf16 by configuration

The OmniVoice deploy file vllm-omni ships sets `dtype: float32`, and the model
runs in fp32. Setting `dtype: bfloat16` is not enough on stock 0.28.0: the bf16
path builds an attention mask in fp32 and passes it to a bf16 query, and the
forward pass fails on the dtype mismatch.

**Effect on the 0.28.0 measurements.** OmniVoice's only change then, bf16,
needed the library, so the two steps could not be separated either. Measured
together against the fp32 default they were 6.07x / 6.45x, of which bf16 alone
was 4.04x and the rest request batching.
`omnivoice` in this repository carries `dtype: bfloat16`, the configuration
measured on 0.30.0; on 0.28.0 it needs the mask fix underneath it to run.

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

**Effect on you.** `voxcpm2` pins `max_num_seqs: 32` and
`unified_decode_graph_max_batch_size: 32` unconditionally, which is the H100 80GB
point. On a smaller card serve vllm-omni's own `voxcpm2.yaml` instead: the pinned pair will
try to admit 32 requests regardless of how much memory the device has. The two
keys have to move together -- raising admission alone leaves every batch above
the graph limit off the unified decode graph, so the extra admission buys
nothing.

## General

- **`connectors:` is replaced, not merged, by a 0.28.0 overlay.** An overlay
  that changes one streaming key must restate the whole `connectors:` block; a
  key left out reverts to its class default rather than to the base's value.
  This has silently changed `decode_batch_max_size` from the base's 4 to the code
  default of unlimited in configs that forgot it. The configs here are
  self-contained, with no `base_config`, so this no longer applies to them.
- **`decode_cudagraph_capture_sizes` is ignored under `async_chunk` on 0.28.0.**
  The decoder derives its capture shapes from the chunk settings. The key is
  still present in the shipped configs so they read as the configurations they
  were measured as; do not attribute a gain to it on this release.
- **`platforms:` blocks cannot be removed by an overlay.** The loader always
  unions a base's platform entries into the merged result. The Qwen3-TTS configs
  here keep the non-CUDA platform block the measured configuration inherited
  from its base. It is inert on the CUDA path these numbers were taken on.
