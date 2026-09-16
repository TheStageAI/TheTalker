#!/usr/bin/env bash
# Start a vllm-omni TTS server on one of this repository's deploy configs.
#
# Usage:
#   examples/serve.sh <config-name> [port]
#
#   <config-name>  any name `thetalker-deploy` lists, e.g. qwen3tts17b_optimized
#   MODEL=<hf-id>  serve a different checkpoint than the config's own model
#   HOST=<addr>    bind address, default 127.0.0.1; 0.0.0.0 serves the network
#
# Example (Qwen3-TTS 1.7B-Base, optimized serving parameters, port 8091):
#   examples/serve.sh qwen3tts17b_optimized 8091
set -euo pipefail

CONFIG="${1:?usage: serve.sh <config-name> [port]}"
PORT="${2:-8091}"
HOST="${HOST:-127.0.0.1}"

# Ask thetalker-deploy for the checkpoint rather than keeping a second copy of
# the table here: a copy keyed on the config-name prefix missed every alias
# (batch32, tuned), which this script's own usage says it accepts.
if [ -z "${MODEL:-}" ]; then
    MODEL="$(thetalker-deploy --model-id "$CONFIG")" || exit $?
fi

DEPLOY_CONFIG="$(thetalker-deploy "$CONFIG")"
echo "serving $MODEL on $HOST:$PORT with $DEPLOY_CONFIG"

vllm serve "$MODEL" \
    --omni --trust-remote-code --host "$HOST" --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG"
