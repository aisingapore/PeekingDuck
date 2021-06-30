# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code of this file is mostly forked from
# [@xuannianz](https://github.com/xuannianz))

"""
Utility functions to obtain submodules
"""

from typing import Dict, Any, Tuple

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Helper function to get keras submodules

    Args:
        kwargs (dict]): dictionary of keyword args

    Returns:
        (tuple): the specified submodules
    """
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError(f'Invalid keyword argument: {key}')
    return backend, layers, models, utils
