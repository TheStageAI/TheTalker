"""Deploy-config regression + equivalence tests. No server, no GPU.

Three independent checks:

1. ``test_all_deploy_yamls_load`` -- every file under ``thetalker/deploy/`` parses
   via vllm-omni's own loader. Runs whenever ``vllm_omni`` is importable.

2. ``test_materialized_config_loads`` -- what ``thetalker-deploy <name>`` prints
   loads too, in both base-resolution modes, and the packaged default resolves
   to the same DeployConfig as the vllm-omni file it was copied from.

3. ``test_optimized_equals_measured_source`` -- proves that expressing an
   ``*_optimized.yaml`` as a thin overlay on the sibling ``*_default.yaml``
   (``base_config: ..._default.yaml`` plus only the changed keys) produces the
   exact same merged config as the standalone file that the measurement chain
   actually served. That standalone source file is internal and does not ship
   in this repository, so this half of the test needs the
   ``TTS_BENCH_MEASURED_DEPLOY_DIR`` environment variable pointed at a
   directory holding it; without that variable it is skipped, not failed.

All three are no-ops (skipped) when ``vllm_omni`` is not installed.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from thetalker import deploy_configs

vllm_omni_config = pytest.importorskip(
    "vllm_omni.config.stage_config",
    reason="vllm_omni is not installed; deploy-config checks need the vllm-omni package",
)
load_deploy_config = vllm_omni_config.load_deploy_config

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "thetalker" / "deploy"

ALL_DEPLOY_YAMLS = sorted(DEPLOY_DIR.glob("*.yaml"))

# repo overlay filename -> measured-source filename (lives outside this repo;
# see TTS_BENCH_MEASURED_DEPLOY_DIR above). One pair (qwen3tts17b) also
# inherits a `platforms:` block from its sibling `_default.yaml` that the
# standalone measured source does not have -- the loader's base_config merge
# always unions a base's `platforms:` entries into the overlay's, with no way
# for an overlay to remove an inherited platform block. That block only
# activates on non-CUDA platforms (rocm/npu/xpu) and is inert on the H100
# these configs are measured on, so it is excluded from the equality check
# below and asserted separately to come only from the sibling default.
OPTIMIZED_TO_MEASURED_SOURCE = {
    "qwen3tts17b_optimized.yaml": "qwen3tts17b_configonly.yaml",
    "qwen3tts06b_optimized.yaml": "qwen3tts06b_configonly.yaml",
    "moss_optimized.yaml": "moss_tuned.yaml",
    "voxcpm2_optimized.yaml": "voxcpm2_params.yaml",
    "omnivoice_optimized.yaml": "omnivoice_bf16.yaml",
}
INHERITED_PLATFORMS_ONLY = {"qwen3tts17b_optimized.yaml"}


def test_all_deploy_yamls_load():
    assert ALL_DEPLOY_YAMLS, f"no deploy yamls found under {DEPLOY_DIR}"
    for path in ALL_DEPLOY_YAMLS:
        load_deploy_config(path)  # raises on any schema/merge problem


@pytest.mark.parametrize("name", deploy_configs.list_names())
@pytest.mark.parametrize("stock_base", [False, True])
def test_materialized_config_loads(tmp_path, name, stock_base):
    """`thetalker-deploy <name>` prints a file that serves, in both base modes."""
    path = deploy_configs.materialize(name, out_dir=str(tmp_path), stock_base=stock_base)
    assert os.path.isabs(path)
    materialized = dataclasses.asdict(load_deploy_config(path))
    shipped = dataclasses.asdict(load_deploy_config(DEPLOY_DIR / f"{name}.yaml"))
    assert materialized == shipped, (
        f"{name}: resolving base_config with stock_base={stock_base} changed the "
        f"merged config; the packaged default is no longer equal to the vllm-omni file"
    )


@pytest.mark.parametrize("overlay_name", sorted(OPTIMIZED_TO_MEASURED_SOURCE))
def test_optimized_equals_measured_source(overlay_name):
    measured_dir = os.environ.get("TTS_BENCH_MEASURED_DEPLOY_DIR")
    if not measured_dir:
        pytest.skip(
            "TTS_BENCH_MEASURED_DEPLOY_DIR not set; the measured source yamls "
            "are internal and are not shipped in this repository"
        )
    source_path = Path(measured_dir) / OPTIMIZED_TO_MEASURED_SOURCE[overlay_name]
    if not source_path.is_file():
        pytest.skip(f"measured source not found: {source_path}")

    overlay_cfg = dataclasses.asdict(load_deploy_config(DEPLOY_DIR / overlay_name))
    source_cfg = dataclasses.asdict(load_deploy_config(source_path))

    if overlay_name in INHERITED_PLATFORMS_ONLY:
        default_name = overlay_name.replace("_optimized.yaml", "_default.yaml")
        default_cfg = dataclasses.asdict(load_deploy_config(DEPLOY_DIR / default_name))
        assert overlay_cfg["platforms"] == default_cfg["platforms"], (
            f"{overlay_name}: inherited platforms block does not match its "
            f"sibling default -- this is not the expected inheritance artifact"
        )
        assert source_cfg["platforms"] is None, (
            f"{overlay_name}: measured source unexpectedly has a platforms "
            f"block; the inherited-platforms exception no longer applies"
        )
        overlay_cfg = {**overlay_cfg, "platforms": None}

    assert overlay_cfg == source_cfg, (
        f"{overlay_name} does not reproduce {OPTIMIZED_TO_MEASURED_SOURCE[overlay_name]}"
    )
