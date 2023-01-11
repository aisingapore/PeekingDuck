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

from torch.utils.data import DataLoader

from src.data_module.base import AbstractDataAdapter


class DataAdapter(AbstractDataAdapter):
    """"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = None
        if cfg.adapter_type == "pytorch":
            self.loader = DataLoader

    def train_dataloader(self, dataset):
        return self.loader(
            dataset,
            **self.cfg.train,
        )

    def valid_dataloader(self, dataset):
        return self.loader(
            dataset,
            **self.cfg.valid,
        )

    def test_dataloader(self, dataset):
        return self.loader(
            dataset,
            **self.cfg.test,
        )
