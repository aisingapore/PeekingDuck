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

from pathlib import Path

import pytest
import yaml

from peekingduck.declarative_loader import ConfigLoader


@pytest.fixture
def configloader():
    config_loader = ConfigLoader(Path.cwd() / "tmp_dir")

    return config_loader


def create_config_yaml(node, data):
    node_type, node_name = node.split(".")
    config_dir = Path.cwd() / "tmp_dir" / "configs"

    node_config_dir = config_dir / node_type
    node_config_dir.mkdir(parents=True)

    full_path = node_config_dir / f"{node_name}.yml"

    with open(full_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


@pytest.mark.usefixtures("tmp_dir")
class TestConfigLoader:
    def test_config_loader_returns_correct_config_file_path(self, configloader):
        node = "type.node"
        # .replace("\\","/") for windows where os.path.join uses "\\"
        file_path = str(configloader._get_config_path(node)).replace("\\", "/")

        ground_truth = str(
            Path.cwd() / "tmp_dir" / "configs" / f"{node.replace('.', '/')}.yml"
        ).replace("\\", "/")

        assert file_path == ground_truth

    def test_config_loader_load_correct_yaml(self, configloader):
        node = "input.test"
        data = {"input": "img", "output": "img"}
        create_config_yaml(node, data)

        config = configloader.get(node)

        for key in data.keys():
            assert data[key] == config[key]

    #
    # Test loading of model mapping system from master_map.yml
    # Here we need to test each of the object detection models individually.
    #
    def test_config_loader_load_mapping_efficientdet(self, configloader):
        """Test mapping system for EfficientDet model is correctly loaded"""
        node_name = "model.efficientdet"
        test_map = configloader._load_mapping(node_name)

        assert test_map["person"] == 0
        assert test_map["parking meter"] == 13
        assert test_map["toothbrush"] == 89

    def test_config_loader_load_mapping_yolo(self, configloader):
        """Test mapping system for Yolo model is correctly loaded"""
        node_name = "model.yolo"
        test_map = configloader._load_mapping(node_name)

        assert test_map["person"] == 0
        assert test_map["parking meter"] == 12
        assert test_map["toothbrush"] == 79

    def test_config_loader_load_mapping_yolox(self, configloader):
        """Test mapping system for YoloX model is correctly loaded"""
        node_name = "model.yolox"
        test_map = configloader._load_mapping(node_name)

        assert test_map["person"] == 0
        assert test_map["airplane"] == 4
        assert test_map["clock"] == 74
        assert test_map["toothbrush"] == 79

    def test_config_loader_obj_det_wildcard_efficientdet(self, configloader):
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

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_obj_det_wildcard_yolo(self, configloader):
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

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    #
    # Test class name to object ID translation for different input cases
    # NB: We are not testing model operations here (that is done under model tests).
    #     But a model is required to obtain a mapping table, and Yolo model was just chosen.
    #     Could also test EfficientDet using its own set of ground truths,
    #     but we would only be testing the translations twice.
    #
    def test_config_loader_change_class_name_to_id_all_text(self, configloader):
        """Test translation on all text inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = ["person", "car", "BUS", "CELL PHONE", "oven"]
        ground_truth = ("detect_ids", [0, 2, 5, 67, 69])

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_all_int(self, configloader):
        """Test translation on all integer inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [0, 1, 2, 3, 5]
        ground_truth = ("detect_ids", [0, 1, 2, 3, 5])

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text(self, configloader):
        """Test translation on heterogenous inputs"""
        node_name = "model.yolo"
        key = "detect_ids"
        val = [4, "bicycle", 10, "LAPTOP", "teddy bear"]
        ground_truth = ("detect_ids", [1, 4, 10, 63, 77])

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text_duplicates(
        self, configloader
    ):
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

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth

    def test_config_loader_change_class_name_to_id_mix_int_text_errors(
        self, configloader
    ):
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

        test_res = configloader.change_class_name_to_id(node_name, key, val)
        assert test_res == ground_truth
