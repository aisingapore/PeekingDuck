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

"""test config script"""
import unittest

from hydra import compose, initialize


def test_with_initialize() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        """
        1. initialize will add config_path the config search path within the context
        2. The module with your configs should be importable.
           it needs to have a __init__.py (can be empty).
        3. THe config path is relative to the file calling initialize (this file)
        config is relative to a module
        """
        cfg = compose(config_name="config", overrides=["framework=tensorflow"])
        assert cfg.framework == "tensorflow"


# Usage in unittest style tests is similar.
class TestWithUnittest(unittest.TestCase):
    """test case for default dataset config"""

    def test_generated_config(self) -> None:
        """test gen config"""
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(
                config_name="config",
                overrides=["data_module.dataset.root_dir=data"],
            )

            assert cfg.data_module.dataset == {
                "download": True,
                "url": "https://github.com/gao-hongnan/peekingduck-trainer/releases/download/v0.0.1-alpha/cifar10.zip",
                "blob_file": "cifar10.zip",
                "root_dir": "data",
                "train_dir": "./${.root_dir}/${project_name}",
                "test_dir": "./${.root_dir}/${project_name}",
                "train_csv": "./${.root_dir}/${project_name}/train.csv",
                "image_path_col_name": "image_path",
                "target_col_name": "class_name",
                "target_col_id": "class_id",
                "stratify_by": "${.target_col_name}",
                "classification_type": "multiclass",
                "image_size": 224,
                "num_classes": 10,
                "class_name_to_id": {
                    "airplane": 0,
                    "automobile": 1,
                    "bird": 2,
                    "cat": 3,
                    "deer": 4,
                    "dog": 5,
                    "frog": 6,
                    "horse": 7,
                    "ship": 8,
                    "truck": 9,
                },
                "classes": [
                    "airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                ],
            }
