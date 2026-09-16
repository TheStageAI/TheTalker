"""`thetalker-deploy` name resolution. No server, no GPU, no vllm-omni needed."""

from __future__ import annotations

import os

import pytest

from pathlib import Path

from thetalker import deploy_configs

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "thetalker" / "deploy"


def test_one_pair_per_model():
    names = deploy_configs.list_names()
    shipped = sorted(path.stem for path in DEPLOY_DIR.glob("*.yaml"))
    assert sorted(names) == shipped, "a shipped yaml has no name, or a name has no yaml"
    assert len(set(names)) == len(names)
    for model in deploy_configs.MODELS:
        assert f"{model.key}_default" in names
        assert f"{model.key}_optimized" in names


@pytest.mark.parametrize("name", deploy_configs.list_names())
def test_every_name_materializes(tmp_path, name):
    path = deploy_configs.materialize(name, out_dir=str(tmp_path))
    assert os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    for line in text.splitlines():
        if line.startswith("base_config:"):
            base = line.split(":", 1)[1].strip()
            assert os.path.isabs(base) and os.path.isfile(base), (
                f"{name}: base_config was not resolved to an existing absolute path"
            )


def test_yaml_suffix_and_aliases_resolve():
    assert deploy_configs.resolve("voxcpm2_optimized.yaml")[1] == "voxcpm2_optimized"
    for alias, target in deploy_configs.ALIASES.items():
        assert deploy_configs.resolve(alias)[1] == target


def test_unknown_name_lists_what_is_available():
    with pytest.raises(FileNotFoundError) as excinfo:
        deploy_configs.resolve("no_such_config")
    assert "qwen3tts17b_optimized" in str(excinfo.value)


@pytest.mark.parametrize("name", ["no_such_config", "chunk75_refonce_predf3f", "batch",
                                  "qwen3tts17b_chunk75_refonce"])
def test_cli_exits_2_on_unknown_name(capsys, name):
    """A library overlay name is what a reader types first; it must not traceback."""
    assert deploy_configs.main([name]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    for expected in ("Available:", "voxcpm2_optimized", "batch32 -> voxcpm2_optimized"):
        assert expected in captured.err


def test_cli_lists_names_and_model_ids(capsys):
    assert deploy_configs.main([]) == 0
    out = capsys.readouterr().out
    for name in deploy_configs.list_names():
        assert name in out
    for model in deploy_configs.MODELS:
        assert model.model_id in out


def test_cli_prints_one_path(tmp_path, capsys):
    assert deploy_configs.main(["--out-dir", str(tmp_path), "qwen3tts17b_optimized"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.splitlines() == [out] and os.path.isfile(out)


def test_cli_prints_the_stream_sample_rate(capsys):
    from thetalker import deploy_configs

    assert deploy_configs.main(["--sample-rate", "voxcpm2_optimized"]) == 0
    assert capsys.readouterr().out.strip() == "48000"
    assert deploy_configs.main(["--sample-rate", "qwen3tts17b_optimized"]) == 0
    assert capsys.readouterr().out.strip() == "24000"
