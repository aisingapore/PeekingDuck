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

from typing import Dict

import pytest

from peekingduck.pipeline.nodes.node import AbstractNode


class ConcreteNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=__name__)

    def run(self, inputs: Dict):
        return {"data1": 1, "data2": 42}


class IncorrectNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=__name__)


@pytest.fixture
def c_node():
    return ConcreteNode({'input': ['img'], 'output': ['int']})

class TestNode():

    def test_node_returns_correct_output(self, c_node):
        results = c_node.run({'input': 1})
        assert results == {"data1": 1, "data2": 42}


    def test_node_init_raises_error(self):
        with pytest.raises(KeyError):
            ConcreteNode({})


    def test_node_gives_correct_inputs(self, c_node):
        results = c_node.inputs
        assert results == ["img"]


    def test_node_gives_correct_outputs(self, c_node):
        results = c_node.outputs
        assert results == ["int"]


    def test_node_no_concrete_run_raises_error(self):
        with pytest.raises(TypeError):
            IncorrectNode({})
