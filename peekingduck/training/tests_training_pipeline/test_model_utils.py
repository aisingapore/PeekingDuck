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

# Copyright (c) Megvii, Inc. and its affiliates.
""" Test data module, data adapter, data loader and dataset
"""

import unittest

import torch
from torch import nn

from src.model.yolox.utils import adjust_status, freeze_module
from src.model.yolox.exp import get_exp


class TestModelUtils(unittest.TestCase):
    """Test model utils"""

    def setUp(self) -> None:
        self.model: nn.Module = get_exp(exp_name="yolox-s").get_model()

    def test_model_state_adjust_status(self) -> None:
        """Test model state adjust status"""
        data = torch.ones(1, 10, 10, 10)
        # use bn since bn changes state during train/val
        model = nn.BatchNorm2d(10)
        prev_state = model.state_dict()

        modes = [False, True]
        results = [True, True]

        # test under train/eval mode
        for mode, result in zip(modes, results):
            with adjust_status(model, training=mode):
                model(data)
            model_state = model.state_dict()
            self.assertTrue(len(model_state) == len(prev_state))
            self.assertEqual(
                result,
                all([torch.allclose(v, model_state[k]) for k, v in prev_state.items()]),
            )

        # test recurrsive context case
        prev_state = model.state_dict()
        with adjust_status(model, training=False):
            with adjust_status(model, training=False):
                model(data)
        model_state = model.state_dict()
        self.assertTrue(len(model_state) == len(prev_state))
        self.assertTrue(
            all([torch.allclose(v, model_state[k]) for k, v in prev_state.items()])
        )

    def test_model_effect_adjust_status(self) -> None:
        """test context effect"""
        self.model.train()
        with adjust_status(self.model, training=False):
            for module in self.model.modules():
                self.assertFalse(module.training)
        # all training after exit
        for module in self.model.modules():
            self.assertTrue(module.training)

        # only backbone set to eval
        self.model.backbone.eval()
        with adjust_status(self.model, training=False):
            for module in self.model.modules():
                self.assertFalse(module.training)

        for name, module in self.model.named_modules():
            if "backbone" in name:
                self.assertFalse(module.training)
            else:
                self.assertTrue(module.training)

    def test_freeze_module(self) -> None:
        """test freeze module"""
        model = nn.Sequential(
            nn.Conv2d(3, 10, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        data = torch.rand(1, 3, 10, 10)
        model.train()
        assert isinstance(model[1], nn.BatchNorm2d)
        before_states = model[1].state_dict()
        freeze_module(model[1])
        model(data)
        after_states = model[1].state_dict()
        self.assertTrue(
            all([torch.allclose(v, after_states[k]) for k, v in before_states.items()])
        )

        # yolox test
        self.model.train()
        for module in self.model.modules():
            self.assertTrue(module.training)

        freeze_module(self.model, "backbone")
        for module in self.model.backbone.modules():
            self.assertFalse(module.training)
        for p in self.model.backbone.parameters():
            self.assertFalse(p.requires_grad)

        for module in self.model.head.modules():
            self.assertTrue(module.training)
        for p in self.model.head.parameters():
            self.assertTrue(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
