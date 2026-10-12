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
#   MPS=1          run the server under an NVIDIA MPS daemon, started here and
#                  stopped when the server exits. The vllm-omni column for
#                  Qwen3-TTS and MOSS-TTS 8B is measured this way.
#
# THESTAGE_VLLM_OMNI defaults to 0, which keeps an installed thestage-vllm-omni
# out of the server. The library takes no deploy file: serve it with
# `vllm-omni serve <model> --omni --trust-remote-code`.
#
# Example (Qwen3-TTS 1.7B CustomVoice, port 8091):
#   MPS=1 examples/serve.sh qwen3tts17b 8091
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

EXTRA_ARGS=()
# MOSS-TTS 8B's first start compiles longer than the default 600 s timeout.
if [ "$(basename "$DEPLOY_CONFIG")" = moss.yaml ]; then
    EXTRA_ARGS=(--init-timeout 1800 --stage-init-timeout 1800)
fi

if [ "${MPS:-0}" = 1 ]; then
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps-pipe}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps-log}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
fi

vllm-omni serve "$MODEL" \
    --omni --trust-remote-code --host "$HOST" --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
