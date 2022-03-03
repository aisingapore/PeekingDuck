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

from typing import Dict

import pytest

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.utils.create_node_helper import obj_det_change_class_name_to_id


class ConcreteNode(AbstractNode):
    def __init__(self, config={}, **kwargs):
        super().__init__(config=config, node_path="input.recorded", **kwargs)

    def run(self, inputs: Dict):
        return {"data1": 1, "data2": 42}


class IncorrectNode(AbstractNode):
    def __init__(self, config):
        super().__init__(config, node_path=__name__)


class ObjDetNodeEfficientDet(AbstractNode):
    def __init__(self, config={}, **kwargs):
        node_name = "model.efficientdet"
        super().__init__(config=config, node_path=node_name, **kwargs)

    def run(self, inputs: Dict):
        return {}


class ObjDetNodeYolo(AbstractNode):
    def __init__(self, config={}, **kwargs):
        node_name = "model.yolo"
        super().__init__(config=config, node_path=node_name, **kwargs)

    def run(self, inputs: Dict):
        return {}


@pytest.fixture
def c_node():
    return ConcreteNode({"input": ["img"], "output": ["int"]})


class TestNode:
    def test_node_returns_correct_output(self, c_node):
        results = c_node.run({"input": 1})
        assert results == {"data1": 1, "data2": 42}

    def test_node_init_takes_empty_dictionary(self):
        ConcreteNode({})
        assert True

    def test_node_init_able_to_override_using_kwargs(self):
        tmp_node = ConcreteNode(input_dir="path_to_input")
        assert tmp_node.config["input_dir"] == "path_to_input"

    def test_node_init_able_to_override_using_dict(self):
        tmp_node = ConcreteNode(config={"input_dir": "path_to_input"})
        assert tmp_node.config["input_dir"] == "path_to_input"

    def test_node_gives_correct_inputs(self, c_node):
        results = c_node.inputs
        assert results == ["img"]

    def test_node_gives_correct_outputs(self, c_node):
        results = c_node.outputs
        assert results == ["int"]

    def test_node_no_concrete_run_raises_error(self):
        with pytest.raises(TypeError):
            IncorrectNode({})

    #
    # Test class name to object ID translation for different input cases
    # NB: We are not testing model operations here (that is done under model tests).
    #     But a model is required to obtain a mapping table, and Yolo model was just chosen.
    #     Could also test EfficientDet using its own set of ground truths,
    #     but we would only be testing the translations twice.
    #
    def test_node_obj_det_wildcard_efficientdet(self):
        """Test object detection wildcard for EfficientDet models"""
        node_name = "model.efficientdet"
        key = "detect_ids"
        val = ["*"]
        # fmt: off
        ground_truth = (
            "detect_ids",
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52,
                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 69, 71, 72,
                73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89
            ]
        )
        # fmt: on

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_obj_det_wildcard_yolo(self):
        """Test object detection wildcard for Yolo models"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = ["*"]
        # fmt: off
        ground_truth = (
            "detect_ids",
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
            ]
        )
        # fmt: on

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_all_text(self):
        """Test translation on all text inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = ["person", "car", "BUS", "CELL PHONE", "oven"]
        ground_truth = ("detect_ids", [0, 2, 5, 67, 69])

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_all_int(self):
        """Test translation on all integer inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [0, 1, 2, 3, 5]
        ground_truth = ("detect_ids", [0, 1, 2, 3, 5])

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text(self):
        """Test translation on heterogenous inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [4, "bicycle", 10, "LAPTOP", "teddy bear"]
        ground_truth = ("detect_ids", [1, 4, 10, 63, 77])

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text_duplicates(self):
        """Test translation with heterogenous inputs including duplicates"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [
            4,
            "bicycle",
            10,
            "laptop",
            "teddy bear",
            "aeroplane",
            63,
            10,
        ]
        ground_truth = ("detect_ids", [1, 4, 10, 63, 77])

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text_errors(self):
        """Test translation with heterogenous inputs including errors"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [
            4,
            "bicycle",
            10,
            "laptop",
            "teddy bear",
            "aeroplane",
            63,
            10,
            "pokemon",
            "scary monster",
        ]
        ground_truth = ("detect_ids", [0, 1, 4, 10, 63, 77])

        test_res = obj_det_change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    #
    # Test object detection node initializers to make sure they convert
    # class names to class IDs correctly
    #
    def test_model_yolo_class_ids(self):
        node_config = {"detect_ids": [1, 2, 3, 4, 5]}
        yolo_node = ObjDetNodeYolo(config=node_config)
        ground_truth = [1, 2, 3, 4, 5]
        assert yolo_node.config["detect_ids"] == ground_truth

    def test_model_yolo_class_names(self):
        node_config = {
            "detect_ids": [
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
            ]
        }
        yolo_node = ObjDetNodeYolo(config=node_config)
        ground_truth = [10, 11, 12, 13, 14]
        assert yolo_node.config["detect_ids"] == ground_truth

    def test_model_yolo_class_names_mixed(self):
        node_config = {
            "detect_ids": [
                1,
                2,
                3,
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                20,
                21,
            ]
        }
        yolo_node = ObjDetNodeYolo(config=node_config)
        ground_truth = [1, 2, 3, 10, 11, 12, 13, 14, 20, 21]
        assert yolo_node.config["detect_ids"] == ground_truth

    def test_model_efficientdet_class_ids(self):
        node_config = {"detect_ids": [1, 2, 3, 4, 5]}
        efficientdet_node = ObjDetNodeEfficientDet(config=node_config)
        ground_truth = [1, 2, 3, 4, 5]
        assert efficientdet_node.config["detect_ids"] == ground_truth

    def test_model_efficientdet_class_names(self):
        node_config = {
            "detect_ids": [
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
            ]
        }
        efficientdet_node = ObjDetNodeEfficientDet(config=node_config)
        ground_truth = [10, 12, 13, 14, 15]
        assert efficientdet_node.config["detect_ids"] == ground_truth

    def test_model_efficientdet_class_names_mixed(self):
        node_config = {
            "detect_ids": [
                1,
                2,
                3,
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                20,
                21,
            ]
        }
        efficientdet_node = ObjDetNodeEfficientDet(config=node_config)
        ground_truth = [1, 2, 3, 10, 12, 13, 14, 15, 20, 21]
        assert efficientdet_node.config["detect_ids"] == ground_truth
