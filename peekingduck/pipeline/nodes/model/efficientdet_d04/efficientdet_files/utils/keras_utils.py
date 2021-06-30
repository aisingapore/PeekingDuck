# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))

"""
Utilities to support Keras in EfficientDet
"""


from typing import Any, Dict, Callable
import functools
import tensorflow.keras as tfkeras
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files \
    import efficientnet as model


def init_tfkeras_custom_objects() -> None:
    """Helper function to initialize custom keras objects
    """
    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


def inject_tfkeras_modules(func: Callable) -> Callable:
    """Helper function to wrap input function
    """
    @functools.wraps(func)
    def wrapper(*args: Dict[str, Any], **kwargs: Dict[str, Any]) -> Callable:
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper
