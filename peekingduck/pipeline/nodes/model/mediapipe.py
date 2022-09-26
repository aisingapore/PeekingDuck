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

"""MediaPipe ML solutions."""

from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.model.mediapipev1 import (
    object_detection,
    pose_estimation,
)


class Node(ThresholdCheckerMixin, AbstractNode):
    """Initializes and uses MediaPipe to infer image frame.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:
        task (:obj:`str`): **{"object_detection"},
            default="object_detection"** |br|
            Defines the computer vision task of the model.
        subtask (:obj:`str`): **Refer to CLI command, default=null**. |br|
            Defines the subtask of MediaPipe model.
        model_type (:obj:`str`): **Refer to CLI command, default=0**. |br|
            Defines the type of model to be used for the selected subtask.
        score_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Bounding boxes with confidence score below the threshold will be
            discarded.
    """

    model_constructor = {
        "object_detection": object_detection.ObjectDetectionModel,
        "pose_estimation": pose_estimation.PoseEstimationModel,
    }

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.check_valid_choice("task", {"object_detection", "pose_estimation"})

        self.model = self.model_constructor[self.config["task"]](self.config)
        try:
            self.model.post_init()
        except AttributeError:
            pass
        self._finalize_output_keys()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` perform prediction on it.

        The classes of objects to be detected can be specified through the
        `detect` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        image = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)

        # bboxes, bbox_labels, bbox_scores for object_detection
        # bboxes, bbox_labels, keypoints, keypoint_conns, keypoint_scores for pose_estimation
        results = self.model.predict(image)
        bboxes = np.clip(results[0], 0, 1)

        if self.config["task"] == "object_detection":
            return {
                "bboxes": bboxes,
                "bbox_labels": results[1],
                "bbox_scores": results[2],
            }
        return {
            "bboxes": bboxes,
            "bbox_labels": results[1],
            "keypoints": results[2],
            "keypoint_conns": results[3],
            "keypoint_scores": results[4],
        }

    def _finalize_output_keys(self) -> None:
        """Updates output keys based on the selected ``task``."""
        self.config["output"] = self.config["output"][self.task]
        self.output = self.config["output"]
