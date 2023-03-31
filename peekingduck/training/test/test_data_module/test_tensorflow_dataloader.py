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

""" Test data module, data adapter, data loader and dataset
"""

from typing import List
from pytest import mark

from hydra import compose, initialize
import torch

from src.data.data_adapter import DataAdapter
from src.data.data_module import ImageClassificationDataModule


@mark.parametrize(
    "overrides, expected",
    [
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=tensorflow",
                "debug=True",
                "data_module.dataset.image_size=224",
                "data_module.dataset.download=True",
                "data_module.data_adapter.tensorflow.train.batch_size=32",
                "data_module.data_adapter.tensorflow.validation.batch_size=32",
                "data_module.data_adapter.tensorflow.test.batch_size=32",
            ],
            (32, 224, 224, 3),
        ),
        (
            [
                "project_name=cifar10",
                "data_module=cifar10",
                "framework=pytorch",
                "debug=True",
                "data_module.dataset.image_size=32",
                "data_module.dataset.download=True",
                "data_module.data_adapter.pytorch.train.batch_size=32",
                "data_module.data_adapter.pytorch.validation.batch_size=32",
                "data_module.data_adapter.pytorch.test.batch_size=32",
                "trainer.pytorch.stores.model_artifacts_dir=null",
            ],
            torch.Size([32, 3, 32, 32]),
        ),
    ],
)
def test_data_module(overrides: List[str], expected: List[int]) -> None:
    """Test data_module"""
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

        data_module = ImageClassificationDataModule(
            cfg=cfg.data_module,
        )

        data_module.prepare_data()
        data_module.setup(stage="fit")
        train_loader: DataAdapter = data_module.get_train_dataloader()
        validation_loader: DataAdapter = data_module.get_validation_dataloader()
        assert train_loader
        inputs, _ = next(iter(train_loader))
        assert inputs.shape == expected

        assert validation_loader
        inputs, _ = next(iter(validation_loader))
        assert inputs.shape == expected
