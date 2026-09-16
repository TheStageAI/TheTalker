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
#   WARMUP        requests excluded from stats at the start of each point;
#                 unset, it follows the protocol: 8 at concurrency <= 8, 100
#                 above it. Set it to pin one value for every point.
#   SEED          Poisson/scheduling seed (default 42)
#   CONCURRENCIES space-separated concurrency levels (default "8 32"); a
#                 trailing positional list of concurrencies overrides this
#   TASK_TYPE     request task type: Base | CustomVoice | VoiceDesign; default Base
#   REF_AUDIO     reference wav, required when TASK_TYPE=Base; a clean mono
#                 clip of 5-15 s of the voice to clone (examples/get_reference.py
#                 downloads an openly licensed one)
#   REF_TEXT      transcript of REF_AUDIO, word for word
#   SPEAKER       preset speaker name, required when TASK_TYPE=CustomVoice
#   INSTRUCT      voice description, required when TASK_TYPE=VoiceDesign
#
# Example:
#   python examples/get_reference.py --out reference.wav
#   export REF_AUDIO=reference.wav REF_TEXT="<the transcript it printed>"
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
if [ "$TASK_TYPE" = "Base" ]; then
    EXTRA_ARGS+=(--ref-audio "${REF_AUDIO:?REF_AUDIO is required for --task-type Base}")
    EXTRA_ARGS+=(--ref-text "${REF_TEXT:?REF_TEXT is required for --task-type Base}")
elif [ "$TASK_TYPE" = "CustomVoice" ]; then
    EXTRA_ARGS+=(--speaker "${SPEAKER:?SPEAKER is required for --task-type CustomVoice}")
elif [ "$TASK_TYPE" = "VoiceDesign" ]; then
    EXTRA_ARGS+=(--instruct "${INSTRUCT:?INSTRUCT is required for --task-type VoiceDesign}")
fi

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
        --out-dir "$OUT_DIR/c${c}" \
        "${EXTRA_ARGS[@]}"
done
