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

import logging
from typing import Callable

import tensorflow as tf

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def wrap_frozen_graph(graph_def: tf.compat.v1.GraphDef) -> Callable:
    def _imports_graph_def(img, min_size, factor, thresholds):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
                                                             input_map={'input:0': img,
                                                                        'min_size:0': min_size,
                                                                        'thresholds:0': thresholds,
                                                                        'factor:0': factor},
                                                             return_elements=['prob:0',
                                                                              'landmarks:0',
                                                                              'box:0'],
                                                                               name="")

        return box, prob, landmarks
        
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, 
                                                [tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                                                 tf.TensorSpec(shape=[], dtype=tf.float32),
                                                 tf.TensorSpec(shape=[], dtype=tf.float32),
                                                 tf.TensorSpec(shape=[3], dtype=tf.float32)])

    return wrapped_import

def load_graph(filename: str) -> tf.function:

    with tf.io.gfile.GFile(filename, "rb") as graph_file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_file.read())

        frozen_func = wrap_frozen_graph(graph_def=graph_def)

        return frozen_func