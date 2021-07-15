import pytest
from peekingduck.pipeline.nodes.heuristic.fps import Node


@pytest.fixture
def fps_node():
    node = Node({"input": ["pipeline_end"],
                 "output": ["fps"],
                 "moving_avg": True
                 })
    return node


class TestFPS:
    def test_get_fps(self, fps_node):
        input1 = {"pipeline_end": False}

        assert isinstance(fps_node.run(input1)["fps"], float)

    def test_positive_fps(self, fps_node):
        input1 = {"pipeline_end": False}

        assert fps_node.run(input1)["fps"] > 0
