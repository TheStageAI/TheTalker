"""Packaged vllm-omni deploy configs, two per model: ``_default`` and ``_optimized``.

``_default`` is the file vllm-omni 0.28.0 ships for that model, copied verbatim.
``_optimized`` is a thin overlay on its sibling default carrying only the serving
parameters the benchmark tables were measured with.

An ``_optimized`` file names its base bare (``base_config: voxcpm2_default.yaml``)
because vllm-omni resolves ``base_config`` relative to the overlay file's own
directory.  ``materialize()`` writes a copy with that one line rewritten to an
absolute path, so the printed file is servable from anywhere; pinning an absolute
path in the shipped file would hard-code one machine's site-packages.

CLI:
    thetalker-deploy                          # list names, grouped by model
    thetalker-deploy qwen3tts17b_optimized    # -> path of a ready deploy file
    thetalker-deploy --stock-base voxcpm2_optimized
        resolve the base against the installed vllm-omni deploy dir instead of
        the packaged copy.  The five packaged defaults load to exactly the same
        DeployConfig as vllm-omni 0.28.0's own files, so the two modes agree on
        0.28.0 and diverge only if vllm-omni changes a default later.
    thetalker-deploy --model-id batch32
    thetalker-deploy --sample-rate voxcpm2_optimized  # PCM rate of that model's audio stream
        print the served model id instead of a path, so a wrapper does not have
        to keep its own copy of the name-to-checkpoint table.

Exit status is 2 for an unknown name or a missing vllm-omni, with the reason on
stderr.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from importlib.resources import files

# one token each: resolve() takes the model key off a name by its last "_".
SUFFIXES = ("default", "optimized")


@dataclass(frozen=True)
class Model:
    """One served model and the two deploy configs shipped for it."""

    key: str
    title: str
    model_id: str
    # the vllm-omni deploy filename the pair was copied from / overlays onto.
    stock_base: str
    # PCM rate of the server's audio stream; the clients write WAV headers with it.
    sample_rate: int = 24000


MODELS: tuple[Model, ...] = (
    Model("qwen3tts17b", "Qwen3-TTS 1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3_tts.yaml"),
    Model("qwen3tts06b", "Qwen3-TTS 0.6B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen3_tts.yaml"),
    Model("voxcpm2", "VoxCPM2", "openbmb/VoxCPM2", "voxcpm2.yaml", sample_rate=48000),
    Model("moss", "MOSS-TTS 8B", "OpenMOSS-Team/MOSS-TTS", "moss_tts.yaml"),
    Model("omnivoice", "OmniVoice", "k2-fsa/OmniVoice", "omnivoice.yaml"),
)

# thestage-vllm-omni overlay names that resolve to the same DeployConfig as the
# config here, accepted so one command line works with either package installed.
# The plugin's qwen3_tts and omnivoice overlays also switch on library-only keys,
# so they have no alias: no config in this repository reproduces them.
ALIASES = {
    "batch32": "voxcpm2_optimized",
    "tuned": "moss_optimized",
}


def _data_dir():
    return files("thetalker").joinpath("deploy")


def model_by_key(key: str) -> Model:
    for model in MODELS:
        if model.key == key:
            return model
    known = ", ".join(m.key for m in MODELS)
    raise KeyError(f"Unknown model {key!r}; known models: {known}")


def list_names() -> list[str]:
    """Every config name, in model order, default before optimized."""
    return [f"{model.key}_{suffix}" for model in MODELS for suffix in SUFFIXES]


def resolve(name: str) -> tuple[Model, str]:
    """Map a CLI name to ``(Model, config name)``. Accepts an alias or a ``.yaml`` suffix."""
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    name = ALIASES.get(name, name)
    if name not in list_names():
        raise FileNotFoundError(unknown_name_message(name))
    return model_by_key(name.rsplit("_", 1)[0]), name


def unknown_name_message(name: str) -> str:
    lines = [f"Unknown deploy config {name!r}.", "", "Available:"]
    lines += [f"  {available}" for available in list_names()]
    lines += ["", "thestage-vllm-omni overlay names accepted as aliases:"]
    lines += [f"  {alias} -> {target}" for alias, target in sorted(ALIASES.items())]
    lines += [
        "",
        "The library's other overlay names (chunk75_refonce_predf3f, chunk75,",
        "chunk50, predf3f, batch) switch on library-only keys, so no config here",
        "reproduces them.",
    ]
    return "\n".join(lines)


def _stock_deploy_dir() -> str:
    # find_spec locates the installed package without executing it -- importing
    # vllm_omni drags in vllm and prints log lines, which would end up inside
    # --deploy-config "$(thetalker-deploy <name>)".
    from importlib.util import find_spec

    spec = find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("vllm_omni is not installed")
    return os.path.join(list(spec.submodule_search_locations)[0], "deploy")


def default_out_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "thetalker", "deploy")


def materialize(name: str, out_dir: str | None = None, stock_base: bool = False) -> str:
    """Write config ``name`` with an absolute ``base_config`` and return its path."""
    model, config_name = resolve(name)
    text = _data_dir().joinpath(f"{config_name}.yaml").read_text(encoding="utf-8")

    if stock_base:
        base = os.path.join(_stock_deploy_dir(), model.stock_base)
    else:
        base = os.path.join(str(_data_dir()), f"{model.key}_default.yaml")
    if re.search(r"(?m)^base_config:", text):
        if not os.path.isfile(base):
            raise FileNotFoundError(f"Base deploy config not found: {base}")
        text = re.sub(r"(?m)^base_config:.*$", f"base_config: {base}", text, count=1)
    elif stock_base:
        return base

    out_dir = default_out_dir() if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config_name}.yaml")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return out_path


def _print_list() -> None:
    for model in MODELS:
        print(f"{model.title}  ({model.model_id})")
        for suffix in SUFFIXES:
            print(f"  {model.key}_{suffix}")
    print("\nthestage-vllm-omni overlay names accepted as aliases:")
    for alias, target in sorted(ALIASES.items()):
        print(f"  {alias} -> {target}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    out_dir = None
    if "--out-dir" in argv:
        i = argv.index("--out-dir")
        if i + 1 >= len(argv):
            print("--out-dir needs a directory", file=sys.stderr)
            return 2
        out_dir = argv[i + 1]
        del argv[i : i + 2]
    stock_base = "--stock-base" in argv
    model_id = "--model-id" in argv
    sample_rate = "--sample-rate" in argv
    argv = [arg for arg in argv if arg not in ("--stock-base", "--model-id", "--sample-rate")]

    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if not argv or argv[0] in ("-l", "--list"):
        _print_list()
        return 0
    try:
        if model_id:
            answer = resolve(argv[0])[0].model_id
        elif sample_rate:
            answer = resolve(argv[0])[0].sample_rate
        else:
            answer = materialize(argv[0], out_dir=out_dir, stock_base=stock_base)
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        # An unknown name and a missing vllm-omni both already carry a message
        # that says what to do; a traceback would bury it.
        print(exc, file=sys.stderr)
        return 2
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
