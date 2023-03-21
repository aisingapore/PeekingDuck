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
"""Tensorflow callbacks"""

from typing import List, Optional, Union

from omegaconf import DictConfig
import tensorflow as tf


class TensorFlowCallbacksAdapter:
    """Adapter for Tensorflow callbacks"""

    def get_callback(
        self,
        callback_name: str,
        parameters: Optional[Union[DictConfig, dict, str]] = None,
    ) -> tf.keras.callbacks.Callback:
        """Instantiate single callback"""
        return (
            getattr(tf.keras.callbacks, callback_name)(**parameters)
            if parameters is not None and len(parameters) > 0
            else getattr(tf.keras.callbacks, callback_name)
        )

    def get_callbacks(
        self, callbacks: List[Union[DictConfig, dict, str]]
    ) -> List[tf.keras.callbacks.Callback]:
        """Instantiate callbacks from cfg"""
        callbacks_list = []
        for callback in callbacks:
            try:
                if isinstance(callback, DictConfig):
                    for cbkey, cbval in callback.items():
                        callbacks_list.append(self.get_callback(str(cbkey), cbval))
                elif isinstance(callback, str):
                    callbacks_list.append(self.get_callback(callback))
                else:
                    raise TypeError("Invalid callback type")
            except NotImplementedError:
                raise NotImplementedError  # pylint: disable=raise-missing-from

        return callbacks_list
