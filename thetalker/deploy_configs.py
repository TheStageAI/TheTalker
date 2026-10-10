"""Packaged vllm-omni deploy configs, one per model.

Each file is the optimal configuration for stock vllm-omni 0.30.0, the one the
vllm-omni column of the benchmark tables was measured with. It is
self-contained (no ``base_config``), so it does not depend on the deploy
directory of whichever vllm-omni is installed. thestage-vllm-omni needs none of
them: it serves its own configuration when no ``--deploy-config`` is passed.

CLI:
    thetalker-deploy                       # list names, one per model
    thetalker-deploy qwen3tts17b           # -> path of that model's deploy file
    thetalker-deploy --model-id batch32
    thetalker-deploy --sample-rate voxcpm2 # PCM rate of that model's audio stream
        print the served model id instead of a path, so a wrapper does not have
        to keep its own copy of the name-to-checkpoint table.

Exit status is 2 for an unknown name, with the reason on stderr.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Model:
    """One served model and its deploy config."""

    key: str
    title: str
    model_id: str
    # the vllm-omni 0.30.0 deploy file the config's header compares against.
    stock_config: str
    # PCM rate of the server's audio stream; the clients write WAV headers with it.
    sample_rate: int = 24000


MODELS: tuple[Model, ...] = (
    Model("qwen3tts17b", "Qwen3-TTS 1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen3_tts.yaml"),
    Model("qwen3tts06b", "Qwen3-TTS 0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3_tts.yaml"),
    Model("voxcpm2", "VoxCPM2", "openbmb/VoxCPM2", "voxcpm2.yaml", sample_rate=48000),
    Model("moss", "MOSS-TTS 8B", "OpenMOSS-Team/MOSS-TTS", "moss_tts.yaml"),
    Model("omnivoice", "OmniVoice", "k2-fsa/OmniVoice", "omnivoice.yaml"),
)

# thestage-vllm-omni overlay names that resolve to the same DeployConfig as the
# config here, accepted so one command line works with either package installed.
# The plugin's qwen3_tts and omnivoice overlays also switch on library-only keys,
# so they have no alias: no config in this repository reproduces them.
ALIASES = {
    "batch32": "voxcpm2",
    "tuned": "moss",
}


def _data_dir():
    return files("thetalker").joinpath("deploy")


def list_names() -> list[str]:
    """Every config name, in model order."""
    return [model.key for model in MODELS]


def resolve(name: str) -> Model:
    """Map a CLI name to its Model. Accepts an alias or a ``.yaml`` suffix."""
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    name = ALIASES.get(name, name)
    for model in MODELS:
        if model.key == name:
            return model
    raise FileNotFoundError(unknown_name_message(name))


def unknown_name_message(name: str) -> str:
    lines = [f"Unknown deploy config {name!r}.", "", "Available:"]
    lines += [f"  {available}" for available in list_names()]
    lines += ["", "thestage-vllm-omni overlay names accepted as aliases:"]
    lines += [f"  {alias} -> {target}" for alias, target in sorted(ALIASES.items())]
    lines += [
        "",
        "The library's other overlay names (chunk75_init16, chunk75_init16_mk, rtfx,",
        "ttfa, chunk75_refonce_predf3f, chunk75, chunk50, predf3f, batch) switch on",
        "library-only keys, so no config here reproduces them. The library serves",
        "its own configuration when no --deploy-config is passed.",
    ]
    return "\n".join(lines)


def config_path(name: str) -> str:
    """Absolute path of the packaged deploy file for ``name``."""
    path = _data_dir().joinpath(f"{resolve(name).key}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Packaged deploy config not found: {path}")
    return str(path)


def _print_list() -> None:
    for model in MODELS:
        print(f"{model.key:<12} {model.title}  ({model.model_id})")
    print("\nthestage-vllm-omni overlay names accepted as aliases:")
    for alias, target in sorted(ALIASES.items()):
        print(f"  {alias} -> {target}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    model_id = "--model-id" in argv
    sample_rate = "--sample-rate" in argv
    argv = [arg for arg in argv if arg not in ("--model-id", "--sample-rate")]

    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if not argv or argv[0] in ("-l", "--list"):
        _print_list()
        return 0
    try:
        if model_id:
            answer = resolve(argv[0]).model_id
        elif sample_rate:
            answer = resolve(argv[0]).sample_rate
        else:
            answer = config_path(argv[0])
    except (FileNotFoundError, OSError) as exc:
        # An unknown name already carries a message that says what to do; a
        # traceback would bury it.
        print(exc, file=sys.stderr)
        return 2
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
