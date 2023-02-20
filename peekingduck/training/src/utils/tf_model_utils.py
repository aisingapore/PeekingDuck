from typing import List

import tensorflow as tf


def set_trainable_layers(model, trainable_layer_name_list: List[str]) -> None:
    """Set the layers in the model to be trainable.
    If a layer is not included in the list, it will be freezed.
    """
    # Needed because if model.trainable = False, the layer.trainable = True will not take effect.
    # Reference: https://github.com/tensorflow/tensorflow/issues/29535
    # model.trainable = True # comment out for testing first. Remove later if not needed,

    # freeze the layers that are not in the provided layer list
    for layer in model.layers:
        if layer.name not in trainable_layer_name_list:
            layer.trainable = False
        else:
            layer.trainable = True

    # confirm the trainable layers
    trainable_layer_list = [
        layer.name for layer in model.layers if layer.trainable == True
    ]
    print(f"\nThe trainable layers are {trainable_layer_list}\n")


# def set_fine_tune_layers(
#     model, prediction_layer_name: str, trainable_layer_name_list: List[str]
# ):
#     trainable_layer_name_list.append(prediction_layer_name)
#     set_trainable_layers(model, trainable_layer_name_list)
