import pytest

from peekingduck.pipeline.nodes.node import AbstractNode

from typing import Dict


class ConcreteNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='ConcreteNode')

    def run(self, inputs: Dict):
        return {"data1": 1, "data2": 42}

class IncorrectNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='ConcreteNode')

c_node = ConcreteNode({'input': ['img'], 'output': ['int']})

def test_node_returns_correct_output():
    results = c_node.run({'input': 1})
    assert results == {"data1": 1, "data2": 42}

def test_node_init_raises_error():
    with pytest.raises(KeyError):
        ConcreteNode({})

def test_node_gives_correct_inputs():
    results = c_node.inputs
    assert results == ["img"]

def test_node_gives_correct_outputs():
    results = c_node.outputs
    assert results == ["int"]

def test_node_name_is_correct():
    results = c_node.name
    assert results == 'ConcreteNode'

def test_incorrect_instantiation():
    with pytest.raises(TypeError):
        IncorrectNode()