from typing import Dict

import torch
from omegaconf import DictConfig, ListConfig


def set_trainable_layers(model, trainable_module_name_dict: DictConfig) -> None:
    """
    Set the trainable layers within a pytorch model.
    The model can be a backbone or a complete model.
    """
    for module_name, trainable_layers in trainable_module_name_dict.items():
        # change the trainable state within the module based on the value type
        if type(trainable_layers) is int:
            for param in getattr(model, module_name)[-trainable_layers:].parameters():
                param.requires_grad = True
        elif type(trainable_layers) is ListConfig:
            # get the module
            module = getattr(model, module_name)
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
