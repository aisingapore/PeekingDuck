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


from typing import Protocol


class DataModule(Protocol):
    """Base class for custom framework agnostic data module."""

    def prepare_data(
        self,
    ) -> None:
        """Post init. Misc function to download, prepare raw data before setup"""

    def setup(self, stage: str) -> None:
        """setup function"""

    def get_train_dataset(self):
        """return training dataset"""

    def get_validation_dataset(self):
        """return validation dataset"""

    def get_test_dataset(self):
        """return test dataset"""


class AbstractDataAdapter(Protocol):
    """Base class for data adapter"""

    def train_dataloader(self, dataset):
        """Adapter for training dataset"""

    def validation_dataloader(self, dataset):
        """Adapter for validation dataset"""

    def test_dataloader(self, dataset):
        """Adapter for test dataset"""
