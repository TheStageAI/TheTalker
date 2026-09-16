"""examples/serve.sh resolves every name thetalker-deploy lists. No server, no GPU.

The script used to key the checkpoint off the config name's prefix, which quietly
rejected all three aliases its own usage promises. It now asks
`thetalker-deploy --model-id`, so this test drives the real script with a stub
`vllm` on PATH and checks the command line it would have run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from thetalker import deploy_configs

REPO = Path(__file__).resolve().parent.parent
SERVE = REPO / "examples" / "serve.sh"

EXPECTED = {name: deploy_configs.resolve(name)[0].model_id
            for name in list(deploy_configs.list_names()) + list(deploy_configs.ALIASES)}


@pytest.fixture(scope="module")
def stub_path(tmp_path_factory):
    """A PATH where `vllm` echoes its arguments and `thetalker-deploy` is ours."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    bin_dir = tmp_path_factory.mktemp("stub-bin")

    (bin_dir / "vllm").write_text('#!/usr/bin/env bash\necho "VLLM $*"\n')
    (bin_dir / "vllm").chmod(0o755)

    # Call the CLI in-process rather than relying on a console script being on
    # PATH, so the test runs from a plain checkout too.
    (bin_dir / "thetalker-deploy").write_text(
        f'#!/usr/bin/env bash\nexec {sys.executable} -c '
        f'"import sys; sys.path.insert(0, r\'{REPO}\'); '
        f'from thetalker.deploy_configs import main; sys.exit(main(sys.argv[1:]))" "$@"\n'
    )
    (bin_dir / "thetalker-deploy").chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def run_serve(stub_path, *args, env_extra=None):
    env = {**os.environ, "PATH": stub_path}
    env.pop("MODEL", None)
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(SERVE), *args], capture_output=True, text=True, env=env, timeout=60
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_name_resolves_to_its_checkpoint(stub_path, name):
    result = run_serve(stub_path, name, "8099")
    assert result.returncode == 0, result.stderr
    assert f"VLLM serve {EXPECTED[name]}" in result.stdout, result.stdout


def test_aliases_reach_the_same_checkpoint_as_their_target(stub_path):
    for alias, target in deploy_configs.ALIASES.items():
        assert EXPECTED[alias] == EXPECTED[target]


def test_unknown_name_fails_without_starting_a_server(stub_path):
    result = run_serve(stub_path, "no_such_config", "8099")
    assert result.returncode != 0
    assert "VLLM" not in result.stdout
    assert "Unknown deploy config" in result.stderr


def test_model_env_overrides_the_checkpoint(stub_path):
    result = run_serve(stub_path, "voxcpm2_optimized", "8099", env_extra={"MODEL": "me/mine"})
    assert result.returncode == 0, result.stderr
    assert "VLLM serve me/mine" in result.stdout


def test_binds_localhost_unless_host_is_set(stub_path):
    result = run_serve(stub_path, "voxcpm2_optimized", "8099")
    assert "--host 127.0.0.1" in result.stdout
    result = run_serve(stub_path, "voxcpm2_optimized", "8099", env_extra={"HOST": "0.0.0.0"})
    assert "--host 0.0.0.0" in result.stdout
