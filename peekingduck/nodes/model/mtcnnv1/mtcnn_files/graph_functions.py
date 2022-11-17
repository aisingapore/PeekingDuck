# Copyright 2022 AI Singapore
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

"""Utility functions to load the MTCNN model."""

import logging
from typing import Callable, List, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def load_graph(file_path: str, inputs: List[str], outputs: List[str]) -> Callable:
    """Loads a frozen graph and wraps it into a function.

    Args:
        file_path (str): Path to the frozen graph or model.
        inputs (List[str]): List of input names.
        outputs (List[str]): List of tensor names to be returned.

    Returns:
        (Callable): A WrappedFunction which wraps a tf V1 piece of code in a
        function.
    """
    with tf.io.gfile.GFile(file_path, "rb") as graph_file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_file.read())

        frozen_func = wrap_frozen_graph(graph_def, inputs, outputs)

        return frozen_func


def wrap_frozen_graph(
    graph_def: tf.compat.v1.GraphDef, inputs: List[str], outputs: List[str]
) -> Callable:
    """Wraps a frozen graph into a function.

    Args:
        graph_def (tf.compat.v1.GraphDef): A frozen graph in graph_def format.
        inputs (List[str]): List of input names.
        outputs (List[str]): List of tensor names to be returned.

    Return:
        (Callable): A wrapped_import function to perform your inference with.
    """

    def _imports_graph_def(
        img: tf.Tensor, min_size: tf.Tensor, factor: tf.Tensor, thresholds: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        prob, landmarks, box = tf.compat.v1.import_graph_def(
            graph_def,
            input_map={
                inputs[0]: img,
                inputs[1]: min_size,
                inputs[2]: thresholds,
                inputs[3]: factor,
            },
            return_elements=outputs,
            name="",
        )

        return box, prob, landmarks

    wrapped_import = tf.compat.v1.wrap_function(
        _imports_graph_def,
        [
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[3], dtype=tf.float32),
        ],
    )

    return wrapped_import
