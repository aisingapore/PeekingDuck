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

"""Trainer Pipeline"""

import logging
from omegaconf import DictConfig
from configs import LOGGER_NAME

logger: logging.Logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name


def run(cfg: DictConfig) -> None:

    if cfg.pipeline == "classification":
        from src.use_case.classification_pipeline import run_classification
        run_classification(cfg)

    elif cfg.pipeline == "detection":
        from src.use_case.detection_pipeline import run_detection
        run_detection(cfg.trainer.yolox)

    # elif cfg.pipeline == "segmentation":
        # from src.use_case.segmentation_pipeline import run_segmentation
        # run_detection(cfg)
