"""Deploy-config regression + equivalence tests. No server, no GPU.

Three independent checks, all skipped when ``vllm_omni`` is not installed:

1. ``test_all_deploy_yamls_load`` -- every file under ``thetalker/deploy/`` parses
   via vllm-omni's own loader.

2. ``test_changes_vs_stock`` -- on vllm-omni 0.30.0, the keys where each shipped
   config differs from that release's own deploy file are exactly the ones its
   header comment and the README list. Skipped on any other vllm-omni version.

3. ``test_config_equals_measured_source`` -- each shipped file loads to the
   exact DeployConfig the measurement chain served. Those measured files are
   internal and do not ship in this repository, so this check needs the
   ``TTS_BENCH_MEASURED_DEPLOY_DIR`` environment variable pointed at a directory
   holding them; without that variable it is skipped, not failed.
"""

from __future__ import annotations

import dataclasses
import json
import os
from importlib.metadata import version
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

# model key -> measured-source filename (lives outside this repo; see
# TTS_BENCH_MEASURED_DEPLOY_DIR above).
MEASURED_SOURCE = {model.key: f"{model.key}_optimized.yaml" for model in deploy_configs.MODELS}

QWEN_CHANGES = {
    "connectors.connector_of_shared_memory.extra.codec_chunk_frames",
    "connectors.connector_of_shared_memory.extra.decode_cudagraph_batch_sizes",
    "connectors.connector_of_shared_memory.extra.initial_codec_chunk_frames",
    "connectors.connector_of_shared_memory.extra.ref_code_context_frames",
    "model_runner",
    "platforms.musa",
    "platforms.npu.model_runner",
    "platforms.rocm.model_runner",
    "platforms.xpu",
    "stages[0].max_num_batched_tokens",
    "stages[1].engine_extras.dtype",
}
CHANGES_VS_STOCK_030 = {
    "qwen3tts17b": QWEN_CHANGES,
    "qwen3tts06b": QWEN_CHANGES,
    "voxcpm2": {
        "stages[0].max_num_seqs",
        "stages[0].engine_extras.hf_overrides.voxcpm2_runtime_config."
        "unified_decode_graph_max_batch_size",
    },
    "moss": {
        "platforms",
        "stages[0].max_num_seqs",
        "stages[0].gpu_memory_utilization",
        "stages[1].max_num_seqs",
        "stages[1].gpu_memory_utilization",
        "stages[1].max_num_batched_tokens",
    },
    "omnivoice": {
        "stages[0].engine_extras.dtype",
        "stages[0].engine_extras.request_batch_max_wait_ms",
        "stages[0].max_num_seqs",
    },
}


def _as_plain(path) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(load_deploy_config(path)), default=str))


def _changed_paths(a, b, prefix=""):
    """Dotted paths where two loaded configs differ, descending into dicts and stage lists."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for key in set(a) | set(b):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.add(child)
            else:
                out |= _changed_paths(a[key], b[key], child)
        return out
    if (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
            and all(isinstance(x, dict) for x in a + b)):
        out = set()
        for i, (x, y) in enumerate(zip(a, b)):
            out |= _changed_paths(x, y, f"{prefix}[{i}]")
        return out
    return set() if a == b else {prefix}


def test_all_deploy_yamls_load():
    assert ALL_DEPLOY_YAMLS, f"no deploy yamls found under {DEPLOY_DIR}"
    for path in ALL_DEPLOY_YAMLS:
        load_deploy_config(path)  # raises on any schema problem


@pytest.mark.parametrize("model", deploy_configs.MODELS, ids=lambda m: m.key)
def test_changes_vs_stock(model):
    if version("vllm-omni") != "0.30.0":
        pytest.skip("the recorded changes are against vllm-omni 0.30.0's deploy files")
    import vllm_omni

    stock = Path(vllm_omni.__file__).parent / "deploy" / model.stock_config
    changed = _changed_paths(_as_plain(stock), _as_plain(DEPLOY_DIR / f"{model.key}.yaml"))
    assert changed == CHANGES_VS_STOCK_030[model.key]


@pytest.mark.parametrize("key", sorted(MEASURED_SOURCE))
def test_config_equals_measured_source(key):
    measured_dir = os.environ.get("TTS_BENCH_MEASURED_DEPLOY_DIR")
    if not measured_dir:
        pytest.skip(
            "TTS_BENCH_MEASURED_DEPLOY_DIR not set; the measured source yamls "
            "are internal and are not shipped in this repository"
        )
    source_path = Path(measured_dir) / MEASURED_SOURCE[key]
    if not source_path.is_file():
        pytest.skip(f"measured source not found: {source_path}")

    assert _as_plain(DEPLOY_DIR / f"{key}.yaml") == _as_plain(source_path), (
        f"{key}.yaml does not reproduce {MEASURED_SOURCE[key]}"
    )
