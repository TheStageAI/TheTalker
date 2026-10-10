"""`thetalker-deploy` name resolution. No server, no GPU, no vllm-omni needed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from thetalker import deploy_configs

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "thetalker" / "deploy"


def test_one_config_per_model():
    names = deploy_configs.list_names()
    shipped = sorted(path.stem for path in DEPLOY_DIR.glob("*.yaml"))
    assert sorted(names) == shipped, "a shipped yaml has no name, or a name has no yaml"
    assert len(set(names)) == len(names)
    assert names == [model.key for model in deploy_configs.MODELS]


@pytest.mark.parametrize("name", deploy_configs.list_names())
def test_every_name_is_a_self_contained_file(name):
    path = deploy_configs.config_path(name)
    assert os.path.isabs(path) and os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    assert not any(line.startswith("base_config:") for line in text.splitlines()), (
        f"{name}: a shipped config must not depend on another file"
    )


def test_yaml_suffix_and_aliases_resolve():
    assert deploy_configs.resolve("voxcpm2.yaml").key == "voxcpm2"
    for alias, target in deploy_configs.ALIASES.items():
        assert deploy_configs.resolve(alias).key == target


def test_unknown_name_lists_what_is_available():
    with pytest.raises(FileNotFoundError) as excinfo:
        deploy_configs.resolve("no_such_config")
    assert "qwen3tts17b" in str(excinfo.value)


@pytest.mark.parametrize("name", ["no_such_config", "chunk75_refonce_predf3f", "batch",
                                  "qwen3tts17b_optimized", "rtfx", "ttfa"])
def test_cli_exits_2_on_unknown_name(capsys, name):
    """A library overlay name is what a reader types first; it must not traceback."""
    assert deploy_configs.main([name]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    for expected in ("Available:", "voxcpm2", "batch32 -> voxcpm2", "no --deploy-config"):
        assert expected in captured.err


def test_cli_lists_names_and_model_ids(capsys):
    assert deploy_configs.main([]) == 0
    out = capsys.readouterr().out
    for name in deploy_configs.list_names():
        assert name in out
    for model in deploy_configs.MODELS:
        assert model.model_id in out


def test_cli_prints_one_path(capsys):
    assert deploy_configs.main(["qwen3tts17b"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.splitlines() == [out] and os.path.isfile(out)
    assert out.endswith("qwen3tts17b.yaml")


def test_cli_prints_the_stream_sample_rate(capsys):
    assert deploy_configs.main(["--sample-rate", "voxcpm2"]) == 0
    assert capsys.readouterr().out.strip() == "48000"
    assert deploy_configs.main(["--sample-rate", "qwen3tts17b"]) == 0
    assert capsys.readouterr().out.strip() == "24000"
