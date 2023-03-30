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

"""test training pipeline script"""
from typing import List
from pytest import mark

from hydra import compose, initialize

import src.training_pipeline


# Drives some user logic with the composed config.
# In this case it calls peekingduck.training.main.add(), passing it the composed config.
@mark.parametrize(
    "overrides, validation_loss_key, expected",
    [
        (["framework=tensorflow"], "val_loss", 2.2),
        (
            ["framework=pytorch", "trainer.pytorch.stores.model_artifacts_dir=null"],
            "valid_loss",
            2.2,
        ),
    ],
)
def test_user_logic(
    overrides: List[str], validation_loss_key: str, expected: float
) -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )
        history = src.training_pipeline.run(cfg)
        assert history is not None
        assert validation_loss_key in history
        assert len(history[validation_loss_key]) != 0
        assert history[validation_loss_key][-1] < expected
