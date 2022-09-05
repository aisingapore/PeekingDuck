# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Util functions using Hugging Face Hub API."""

from typing import Set

from huggingface_hub import hf_api


def get_valid_models(task: str) -> Set[str]:
    """Returns a set containing valid model names for the specified ``task``.

    For ``object_detection``, non-transformers and owlvit models are filtered.

    Args:
        task (str): Computer vision task, e.g., "object_detection".

    Returns:
        (Set[str]): A set of valid Hugging Face Hub models.
    """
    pkd_to_hf_task = {"object_detection": "object-detection"}
    is_valid_model = {"object_detection": is_valid_object_detection_model}

    model_infos = hf_api.list_models(filter=pkd_to_hf_task[task])

    return {info.modelId for info in model_infos if is_valid_model[task](info)}


def is_valid_object_detection_model(model_info: hf_api.ModelInfo) -> bool:
    """True if model is usable with ``transformers`` and is not from the
    owlvit family.
    """
    return (
        "transformers" in model_info.tags
        and "owlvit" not in model_info.tags
        and not any("gpl" in tag for tag in model_info.tags if "license:" in tag)
    )
