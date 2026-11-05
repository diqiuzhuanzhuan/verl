# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Load Mistral Small 4's packed native FP8 experts as trainable dense weights."""


def mistral4_fp8_load_kwargs(hf_config):
    """Supply FP8 expert conversions absent from Transformers 5.16.1.

    The generic FP8 loader only consumes ``*.weight`` and ``*.weight_scale_inv``.
    Mistral's packed experts instead use ``gate_up_proj[_scale_inv]`` and
    ``down_proj[_scale_inv]``; both must be collected before dequantization.
    """
    if getattr(getattr(hf_config, "text_config", None), "model_type", None) != "mistral4":
        return {}
    quantization = getattr(hf_config, "quantization_config", None)
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if not quantization or quantization.get("quant_method") != "fp8" or not quantization.get("dequantize"):
        return {}

    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    class PackedExpertDequantize(Fp8Dequantize):
        def convert(self, input_dict, target_patterns, **kwargs):
            converted = super().convert(input_dict, **kwargs)
            # Remove the source regex's end anchor so WeightConverter can
            # expand this suffix back to the fully qualified parameter name.
            return {target_patterns[0]: next(iter(converted.values()))}

    if get_checkpoint_conversion_mapping("mistral4") is None:
        register_checkpoint_conversion_mapping(
            "mistral4",
            [
                WeightConverter(
                    source_patterns=[
                        f"mlp.experts.{projection}$",
                        f"mlp.experts.{projection}_scale_inv$",
                        f"mlp.experts.{projection}_activation_scale$",
                    ],
                    target_patterns=f"mlp.experts.{projection}",
                    # This op only needs the model/parameter dtype in the load
                    # direction. No quantizer is used to re-quantize training weights.
                    operations=[PackedExpertDequantize(None)],
                )
                for projection in ("gate_up_proj", "down_proj")
            ],
        )
    return {
        "key_mapping": {
            r"^language_model\.model\.": "model.language_model.",
            r"^language_model\.lm_head\.": "lm_head.",
        }
    }
