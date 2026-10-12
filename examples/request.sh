#!/usr/bin/env bash
# One streaming TTS request against a running server (see examples/serve.sh),
# saved as raw PCM. Requires a reference clip + its transcript for the Base
# (voice-clone) task type: a clean mono wav of 5-15 s of the voice to clone,
# plus its word-for-word transcript. examples/get_reference.py downloads an
# openly licensed one.
#
# Usage:
#   examples/request.sh <ref-audio.wav> "<reference transcript>" "<text to speak>" [host:port]
set -euo pipefail

REF_AUDIO="${1:?usage: request.sh <ref-audio.wav> <ref-text> <text> [host:port]}"
REF_TEXT="${2:?usage: request.sh <ref-audio.wav> <ref-text> <text> [host:port]}"
TEXT="${3:?usage: request.sh <ref-audio.wav> <ref-text> <text> [host:port]}"
HOST_PORT="${4:-127.0.0.1:8091}"

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# The body is piped, not passed as an argument: a 10 s reference wav is ~600 KB
# of base64 and the kernel caps a single argument at 128 KB, so `-d "{...}"`
# with the clip inlined fails with "Argument list too long".
{
    printf '{"input": "%s", "task_type": "Base", "language": "English", ' \
        "$(json_escape "$TEXT")"
    printf '"ref_audio": "data:audio/wav;base64,'
    base64 -w0 "$REF_AUDIO"
    printf '", "ref_text": "%s", ' "$(json_escape "$REF_TEXT")"
    printf '"stream": true, "stream_format": "audio", "response_format": "pcm"}'
} | curl -sS "http://${HOST_PORT}/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    --data-binary @- --output out.pcm

echo "wrote out.pcm"
