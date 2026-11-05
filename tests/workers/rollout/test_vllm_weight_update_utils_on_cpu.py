# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _vocab_parallel_embedding_stubs():
    """Stubs standing in for vLLM's ``VocabParallelEmbedding``, or {} if vLLM is real.

    ``weight_update_utils`` imports that class at module scope for tied-alias
    detection, so loading it on CPU needs a placeholder. This file only exercises
    the buffer helpers, which never touch it.
    """
    try:
        importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
    except ImportError:
        pass
    else:
        return {}

    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.vocab_parallel_embedding",
    ]
    stubs = {name: sys.modules.get(name) or types.ModuleType(name) for name in names}
    stubs[names[-1]].VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})
    return stubs


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    stubs = _vocab_parallel_embedding_stubs()
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.configure_fp8_ignored_layers_match_mode = lambda config: None
    fake_vllm_quant.is_quantized_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda weights, runner, is_drafter=False: weights
    fake_vllm_quant.prepare_quanted_weights_for_loading = lambda model: None
    fake_vllm_quant.process_quanted_weights_after_loading = lambda model, state: None

    fake_vllm_unquant = types.ModuleType("verl.utils.vllm.vllm_unquant_utils")
    fake_vllm_unquant.stage_unquantized_moe_params = lambda model: []
    fake_vllm_unquant.fold_unquantized_moe_params = lambda layers: contextlib.nullcontext()

    fake_rocm_expert_map = types.ModuleType("verl.utils.vllm.rocm_vllm_moe_expert_map")
    fake_rocm_expert_map.restore_moe_expert_maps = lambda model: None

    fake_bucketed_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    fake_bucketed_transfer.BucketedWeightReceiver = None

    # NOTE: deliberately do NOT stub verl.plugin.platform. It is lightweight and
    # imports fine on CPU. verl.utils.device binds `get_platform` at import time, so
    # a `lambda: None` stub gets baked into verl.utils.device during this window;
    # because only the keys in `fakes` below are restored, that fake would leak
    # process-wide and later tests would crash in get_device_name() with
    # "'NoneType' object has no attribute 'device_name'".

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.utils.vllm.vllm_unquant_utils": fake_vllm_unquant,
        "verl.utils.vllm.rocm_vllm_moe_expert_map": fake_rocm_expert_map,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucketed_transfer,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.vision_tower.patch_conv.weight", "vision_encoder.patch_conv.weight"),
        ("vision_tower.ln_pre.weight", "vision_encoder.ln_pre.weight"),
        ("model.multi_modal_projector.norm.weight", "pre_mm_projector_norm.weight"),
        ("multi_modal_projector.patch_merger.merging_layer.weight", "patch_merger.merging_layer.weight"),
        ("model.multi_modal_projector.linear_1.weight", "vision_language_adapter.w_in.weight"),
        ("model.multi_modal_projector.linear_2.weight", "vision_language_adapter.w_out.weight"),
        (
            "model.language_model.layers.0.self_attn.q_a_proj.weight_scale_inv",
            "language_model.model.layers.0.self_attn.q_a_proj.weight_scale",
        ),
        (
            "model.language_model.layers.0.mlp.experts.0.gate_proj.activation_scale",
            "language_model.model.layers.0.mlp.experts.0.gate_proj.input_scale",
        ),
        ("language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_pixtral_sync_names_are_normalized_once(name, expected):
    model = type("PixtralForConditionalGeneration", (torch.nn.Module,), {})()
    tensor = torch.ones(2)
    normalize = _weight_update_utils.normalize_pixtral_weight_updates
    result = normalize(model, [(name, tensor)])
    assert result[0][0] == expected and result[0][1] is tensor
    assert normalize(model, result)[0][0] == expected
    other = _ToyModel()
    assert normalize(other, [(name, tensor)])[0][0] == name


def test_pixtral_real_loader_routes_hf_sync_weights_to_each_component(monkeypatch):
    """Use vLLM's real Pixtral load_weights without allocating a GPU model."""
    pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")
    from verl.utils.vllm.vllm_quant_utils import load_quanted_weights

    model = pixtral.PixtralForConditionalGeneration.__new__(pixtral.PixtralForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vision_encoder = torch.nn.Module()
    model.vision_encoder.patch_conv = torch.nn.Linear(2, 2, bias=False)
    model.patch_merger = torch.nn.Module()
    model.patch_merger.merging_layer = torch.nn.Linear(2, 2, bias=False)
    model.pre_mm_projector_norm = torch.nn.LayerNorm(2)
    model.vision_language_adapter = torch.nn.Module()
    model.vision_language_adapter.w_in = torch.nn.Linear(2, 2, bias=False)
    model.vision_language_adapter.w_out = torch.nn.Linear(2, 2, bias=False)
    model.language_model = torch.nn.Module()
    llm_names = []

    def load_language(weights):
        llm_names.extend(name for name, _ in weights)

    model.language_model.load_weights = load_language
    monkeypatch.setattr(_vllm_rollout_utils, "is_quantized_model", lambda _: True)
    monkeypatch.setattr(_vllm_rollout_utils, "load_quanted_weights", load_quanted_weights)
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    worker.model_runner.vllm_config.quant_config = "static_fp8"
    weights = [
        ("model.vision_tower.patch_conv.weight", torch.full((2, 2), 2.0)),
        ("model.multi_modal_projector.patch_merger.merging_layer.weight", torch.full((2, 2), 3.0)),
        ("model.multi_modal_projector.norm.weight", torch.full((2,), 4.0)),
        ("model.multi_modal_projector.linear_1.weight", torch.full((2, 2), 5.0)),
        ("model.multi_modal_projector.linear_2.weight", torch.full((2, 2), 6.0)),
        ("model.language_model.layers.0.self_attn.q_a_proj.weight_scale_inv", torch.tensor([0.25])),
        ("model.language_model.layers.0.self_attn.q_a_proj.activation_scale", torch.tensor([0.5])),
        ("lm_head.weight", torch.ones(2, 2)),
    ]
    worker._update_weights(weights, peft_config=None, base_sync_done=True)
    assert llm_names == [
        "model.layers.0.self_attn.q_a_proj.weight_scale",
        "model.layers.0.self_attn.q_a_proj.input_scale",
        "lm_head.weight",
    ]
    targets = [
        model.vision_encoder.patch_conv.weight,
        model.patch_merger.merging_layer.weight,
        model.pre_mm_projector_norm.weight,
        model.vision_language_adapter.w_in.weight,
        model.vision_language_adapter.w_out.weight,
    ]
    for target, (_, expected) in zip(targets, weights[:5], strict=True):
        torch.testing.assert_close(target, expected)


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)
