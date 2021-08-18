# Copyright 2021 AI Singapore
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

import logging
from typing import List

from peekingduck.pipeline.nodes.draw.utils.general import project_points_onto_original_image


class PackKeypoints:

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def pack(self, model_predictions, image_id, inputs , size):

        for keypoint, score in zip(inputs["keypoints"],
                                   inputs["keypoint_scores"]):

            keypoint = project_points_onto_original_image(keypoint, size)

            hold = []
            for coord in keypoint:
                hold.append(coord[0])
                hold.append(coord[1])
                hold.append(1)

            keypoint = hold

            # self.logger.info(keypoint)
            # self.logger.info(len(keypoint))
            # self.logger.info(score)

            model_predictions.append({"image_id": int(image_id),
                                      "category_id": 1,
                                      "keypoints": list(keypoint),
                                      "score": sum(score)/len(score)})

        return model_predictions