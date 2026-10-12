#!/usr/bin/env bash
# Concurrency sweep against a running server (see examples/serve.sh), one
# benchmark/run_benchmark.py run per concurrency level, output under one
# out-dir per point.
#
# Defaults to the published protocol: 300 requests per point, seed 42, at c8 and
# c32, with the warmup the protocol specifies per point -- 8 requests at c8 and
# 100 at c32. A larger point needs the longer warmup: the codec captures CUDA
# graphs on first use, and at c32 the scored requests must not be the ones
# paying for that.
#
# Usage:
#   examples/run_sweep.sh <out-dir> [host:port] [concurrencies...]
#
# Env overrides:
#   REQUESTS      requests per point (default 300)
#   WARMUP        requests run and drained before each point's scored ones;
#                 unset, it follows the protocol: 8 at concurrency <= 8, 100
#                 above it. Set it to pin one value for every point.
#   SEED          Poisson/scheduling seed (default 42)
#   CONCURRENCIES space-separated concurrency levels (default "8 32"); a
#                 trailing positional list of concurrencies overrides this
#   TASK_TYPE     request task type: Base | CustomVoice | VoiceDesign; default Base
#   REF_AUDIO     reference wav, required when TASK_TYPE=Base; the published
#                 numbers use benchmark/reference/ref.wav
#   REF_TEXT      transcript of REF_AUDIO, sent with it or registered with the
#                 voice; benchmark/reference/ref.txt for the published clip
#   VOICE         name REF_AUDIO is registered under (default thetalker_ref).
#                 Base requests call that registered voice: the first point
#                 uploads it once (POST /v1/audio/voices), every request names it
#   INLINE_REF    1 = send REF_AUDIO inline with every request instead
#   SPEAKER       preset speaker name, required when TASK_TYPE=CustomVoice
#   INSTRUCT      voice description, required when TASK_TYPE=VoiceDesign
#   SAMPLE_RATE   PCM rate of the server's stream (default 24000); VoxCPM2
#                 streams 48000: `thetalker-deploy --sample-rate <name>`
#
# Example:
#   export REF_AUDIO=benchmark/reference/ref.wav
#   export REF_TEXT="$(cat benchmark/reference/ref.txt)"
#   examples/run_sweep.sh /tmp/sweep 127.0.0.1:8091
set -euo pipefail

OUT_DIR="${1:?usage: run_sweep.sh <out-dir> [host:port] [concurrencies...]}"
HOST_PORT="${2:-127.0.0.1:8091}"
shift $(( $# >= 2 ? 2 : $# ))
if [ "$#" -gt 0 ]; then
    CONCURRENCIES=("$@")
elif [ -n "${CONCURRENCIES:-}" ]; then
    read -r -a CONCURRENCIES <<< "$CONCURRENCIES"
else
    CONCURRENCIES=(8 32)
fi

REQUESTS="${REQUESTS:-300}"
SEED="${SEED:-42}"
TASK_TYPE="${TASK_TYPE:-Base}"
HOST="${HOST_PORT%%:*}"
PORT="${HOST_PORT##*:}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_ARGS=(--task-type "$TASK_TYPE")
FIRST_ARGS=()
LATER_ARGS=()
if [ "$TASK_TYPE" = "Base" ]; then
    REF_AUDIO="${REF_AUDIO:?REF_AUDIO is required for --task-type Base}"
    if [ "${INLINE_REF:-0}" = 1 ]; then
        EXTRA_ARGS+=(--ref-audio "$REF_AUDIO")
    else
        VOICE="${VOICE:-thetalker_ref}"
        FIRST_ARGS=(--register-voice "$REF_AUDIO" --voice "$VOICE")
        LATER_ARGS=(--voice "$VOICE")
    fi
    if [ -n "${REF_TEXT:-}" ]; then EXTRA_ARGS+=(--ref-text "$REF_TEXT"); fi
elif [ "$TASK_TYPE" = "CustomVoice" ]; then
    EXTRA_ARGS+=(--speaker "${SPEAKER:?SPEAKER is required for --task-type CustomVoice}")
elif [ "$TASK_TYPE" = "VoiceDesign" ]; then
    EXTRA_ARGS+=(--instruct "${INSTRUCT:?INSTRUCT is required for --task-type VoiceDesign}")
fi

VOICE_ARGS=(${FIRST_ARGS[@]+"${FIRST_ARGS[@]}"})
for c in "${CONCURRENCIES[@]}"; do
    if [ -n "${WARMUP:-}" ]; then
        warmup="$WARMUP"
    elif [ "$c" -le 8 ]; then
        warmup=8
    else
        warmup=100
    fi
    echo "=== concurrency $c (warmup $warmup) ==="
    python "$ROOT/benchmark/run_benchmark.py" \
        --host "$HOST" --port "$PORT" \
        --texts-jsonl "$ROOT/benchmark/prompts/seed_tts_eval_en_100.jsonl" \
        --concurrency "$c" --requests "$REQUESTS" --warmup "$warmup" --seed "$SEED" \
        --out-dir "$OUT_DIR/c${c}" --sample-rate "${SAMPLE_RATE:-24000}" \
        "${EXTRA_ARGS[@]}" ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"}
    VOICE_ARGS=(${LATER_ARGS[@]+"${LATER_ARGS[@]}"})
done
