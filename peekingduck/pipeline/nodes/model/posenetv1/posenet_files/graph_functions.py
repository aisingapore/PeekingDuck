"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import tensorflow as tf

SAVE_DIR = os.path.join(os.getcwd(), 'data', 'posenet')

logger = logging.getLogger(__name__)

def wrap_frozen_graph(graph_def, inputs, outputs):
    '''
    Wraps the graph into a function. This is akin to a model.predict() function
    in keras. When doing inference, simply do frozen_function(tf.cast(x, float))[0].
    It will return your predicted values.

    args:
        - graph_def: The frozen graph in graph_def format
        - inputs: The name(s) of the input nodes from your graph. e.g.['inputs']
        - outputs: The name(s) of your output nodes from your graph.
                    e.g. ['heatmap', 'offsets', 'displacement_fwd', 'displacement_bwd']
        - print_graph: Whether to print the graph

    return:
        a wrapped_import function to perform your inference with.
    '''
    def _imports_graph_def():  # this needs to be here because of graph_def
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def load_graph(filename, inputs, outputs):
    '''
    Loads the graph
    '''
    with tf.io.gfile.GFile(filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        # print_inputs(graph_def)
        # print_outputs(graph_def)

        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=inputs,
                                        outputs=outputs)

        return frozen_func


def print_inputs(graph_def):
    '''
    Prints the input nodes of graph_def
    '''
    # pylint: disable=not-context-manager
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    input_list = []
    for op in graph.get_operations():  # tensorflow.python.framework.ops.Operation
        if op.type == "Placeholder":
            input_list.append(op.name)

        logger.info('Inputs: %s', input_list)


def print_outputs(graph_def):
    '''
    Prints the output nodes of graph_def
    '''
    name_list = []
    input_list = []
    for node in graph_def.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)

    outputs = set(name_list) - set(input_list)
    logger.info('Outputs: %s', list(outputs))
