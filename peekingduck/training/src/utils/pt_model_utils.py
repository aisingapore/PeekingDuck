# Copyright 2023 AI Singapore
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

from omegaconf import DictConfig, ListConfig

from src.model.pytorch_base import PTModel


def unfreeze_all_params(model: PTModel) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_all_params(model: PTModel) -> None:
    for param in model.parameters():
        param.requires_grad = False


def set_trainable_layers(
    model: PTModel, trainable_module_name_dict: DictConfig
) -> None:
    """
    Set the trainable layers within a pytorch model.
    The model can be a backbone or a complete model.
    """
    for module_name, trainable_layers in trainable_module_name_dict.items():
        # change the trainable state within the module based on the value type
        module_name_str: str = str(module_name)
        if isinstance(trainable_layers, int):
            for param in getattr(model, module_name_str)[
                -trainable_layers:
            ].parameters():
                param.requires_grad = True
        elif isinstance(trainable_layers, ListConfig):
            # get the module
            module = getattr(model, module_name_str)
            for layer_name in trainable_layers:
                # get each layer within the module
                layer = getattr(module, layer_name)
                # set trainable for each layer
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(
                f"The config type '{type(trainable_layers)}' for {trainable_layers} is not supported!"
            )
