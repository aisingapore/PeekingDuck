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

    def __init__(self,
            callbacks: List[str] = None,
    ) -> None:
        self.callbacks_list = []
        self.callbacks = callbacks
        for cb in self.callbacks:
            try:
                if type(cb) is DictConfig:
                    for cbkey, cbval in cb.items():
                        self.callbacks_list.append( getattr(self, cbkey)(cbval) )
                elif type(cb) is str:
                    self.callbacks_list.append( getattr(self, cb)() )
                else:
                    raise TypeError
            except NotImplementedError:
                raise NotImplementedError

    # def BackupAndRestore(self, parameters: DictConfig):
    #     return tf.keras.callbacks.BackupAndRestore(**parameters)

    # def BaseLogger(self, parameters: DictConfig):
    #     return tf.keras.callbacks.BaseLogger(**parameters)

    # def CSVLogger(self, parameters: DictConfig):
    #     return tf.keras.callbacks.CSVLogger(**parameters)

    def EarlyStopping(self, parameters: DictConfig = {}):
        return tf.keras.callbacks.EarlyStopping(**parameters)

    def History(self, parameters: DictConfig = {}):
        return tf.keras.callbacks.History(**parameters)

    # def LambdaCallback(self, parameters: DictConfig):
    #     return tf.keras.callbacks.LambdaCallback(**parameters)

    # def LearningRateScheduler(self, parameters: DictConfig):
    #     return tf.keras.callbacks.LearningRateScheduler(**parameters)

    def ModelCheckpoint(self, parameters: DictConfig = {}):
        return tf.keras.callbacks.ModelCheckpoint(**parameters)

    def ProgbarLogger(self, parameters: DictConfig = {}):
        return tf.keras.callbacks.ProgbarLogger(**parameters)

    # def ReduceLROnPlateau(self, parameters: DictConfig):
    #     return tf.keras.callbacks.ReduceLROnPlateau(**parameters)

    # def RemoteMonitor(self, parameters: DictConfig):
    #     return tf.keras.callbacks.RemoteMonitor(**parameters)

    # def TensorBoard(self, parameters: DictConfig):
    #     return tf.keras.callbacks.TensorBoard(**parameters)

    # def TerminateOnNaN(self, parameters: DictConfig):
    #     return tf.keras.callbacks.TerminateOnNaN(**parameters)


    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return self.callbacks_list