#!/usr/bin/env bash
# Start stock vllm-omni on one of this repository's deploy configs (the
# vllm-omni column of the Benchmarks tables).
#
# Usage:
#   examples/serve.sh <config-name> [port]
#
#   <config-name>  any name `thetalker-deploy` lists, e.g. qwen3tts17b
#   MODEL=<hf-id>  serve a different checkpoint than the config's own model
#   HOST=<addr>    bind address, default 127.0.0.1; 0.0.0.0 serves the network
#
# THESTAGE_VLLM_OMNI defaults to 0, which keeps an installed thestage-vllm-omni
# out of the server. The library takes no deploy file: serve it with
# `vllm-omni serve <model> --omni --trust-remote-code`.
#
# Example (Qwen3-TTS 1.7B CustomVoice, port 8091):
#   examples/serve.sh qwen3tts17b 8091
set -euo pipefail

CONFIG="${1:?usage: serve.sh <config-name> [port]}"
PORT="${2:-8091}"
HOST="${HOST:-127.0.0.1}"
export THESTAGE_VLLM_OMNI="${THESTAGE_VLLM_OMNI:-0}"

# Ask thetalker-deploy for the checkpoint rather than keeping a second copy of
# the table here: a copy keyed on the config name missed every alias
# (batch32, tuned), which this script's own usage says it accepts.
if [ -z "${MODEL:-}" ]; then
    MODEL="$(thetalker-deploy --model-id "$CONFIG")" || exit $?
fi

DEPLOY_CONFIG="$(thetalker-deploy "$CONFIG")"
echo "serving $MODEL on $HOST:$PORT with $DEPLOY_CONFIG"

vllm-omni serve "$MODEL" \
    --omni --trust-remote-code --host "$HOST" --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG"
