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

import tensorflow as tf

from typing import List
from omegaconf import DictConfig

class TensorFlowCallbacksAdapter:

    def get_callback(self, callback_name, parameters: DictConfig = {}):
        return getattr(tf.keras.callbacks, callback_name)(**parameters) if len(parameters) > 0 else getattr(tf.keras.callbacks, callback_name)
 

    def get_callbacks(self, callbacks: List[str] = None) -> List[tf.keras.callbacks.Callback]:
        callbacks_list = []
        for cb in callbacks:
            try:
                if type(cb) is DictConfig:
                    for cbkey, cbval in cb.items():
                        callbacks_list.append( self.get_callback(cbkey, cbval) )
                elif type(cb) is str:
                    callbacks_list.append( self.get_callback(cb) )
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError
                
        return callbacks_list