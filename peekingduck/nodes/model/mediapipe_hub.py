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

from typing import Any, Dict, Optional

import cv2
import numpy as np

from peekingduck.nodes.abstract_node import AbstractNode
from peekingduck.nodes.base import ThresholdCheckerMixin
from peekingduck.nodes.model.mediapipe_hubv1 import object_detection, pose_estimation


class Node(ThresholdCheckerMixin, AbstractNode):
    """Initializes and uses MediaPipe to infer image frame.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:
        task (:obj:`str`): **{"object_detection", "pose_estimation"},
            default="object_detection"** |br|
            Defines the computer vision task of the model.
        subtask (:obj:`str`): **Refer to CLI command, default=null**. |br|
            Defines the subtask of MediaPipe model. Use the CLI
            command ``peekingduck model-hub mediapipe tasks`` to obtain a list
            of computer vision tasks and their respective subtasks.
        model_type (:obj:`int`): **Refer to CLI command, default=0**. |br|
            Defines the type of model to be used for the selected subtask. Use
            the CLI command ``peekingduck model-hub mediapipe model-types
            --task '<task>' --subtask '<subtask>'`` to obtain a list of
            supported model types for the specified computer vision tasks and
            subtasks.
        score_threshold (:obj:`float`): **[0, 1], default=0.5**. |br|
            Bounding boxes with confidence score below the threshold will be
            discarded.
        keypoint_format (:obj:`str`): **Refer to CLI command, default="coco"**. |br|
            The expected format of the keypoints output by the model. This could
            be the standard COCO format which is 17 keypoints for the body and
            21 keypoints for the hand. Some models like BlazePose can output 33
            keypoints for the body. Used when ``task = pose_estimation``.
        mirror_image (:obj:`bool`): **default=true**. |br|
            If ``true``, the input image is assumed to be mirrored, i.e., the left
            and right are reversed. Used when ``task = pose_estimation`` and
            ``subtask = hand``.
        static_image_mode (:obj:`bool`): **default=false**. |br|
            If set to ``false``, the solution treats the input images as a video
            stream. It will try to detect the most prominent person in the very
            first images, and upon a successful detection further localizes the
            pose landmarks. In subsequent images, it then simply tracks those
            landmarks without invoking another detection until it loses track,
            on reducing computation and latency. If set to ``true``, person
            detection runs every input image, ideal for processing a batch of
            static, possibly unrelated, images. Used when ``task = pose_estimation``.
        smooth_landmarks (:obj:`bool`): **default=true**. |br|
            If set to ``true``, the solution filters pose landmarks across
            different input images to reduce jitter, but ignored if
            ``static_image_mode`` is also set to ``true``. Used when
            ``task = pose_estimation``.
        tracking_score_threshold (:obj:`float`): **[0, 1], default=0.5**. |br|
            Minimum confidence value from the landmark-tracking model for the
            pose landmarks to be considered tracked successfully, or otherwise
            person detection will be invoked automatically on the next input
            image. Setting it to a higher value can increase robustness of the
            solution, at the expense of a higher latency. Ignored if
            ``static_image_mode = true``, where person detection simply runs on
            every image. Used when ``task = pose_estimation``.
    """

    model_constructor = {
        "object_detection": object_detection.ObjectDetectionModel,
        "pose_estimation": pose_estimation.PoseEstimationModel,
    }

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.check_valid_choice("task", {"object_detection", "pose_estimation"})

        self.model = self.model_constructor[self.config["task"]](self.config)
        self.model.post_init()
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

    def _get_config_types(self) -> Dict[str, Any]:
        return {
            "task": str,
            "model_type": int,
            "score_threshold": float,
            "smooth_landmarks": bool,
            "static_image_mode": bool,
            # Allow `None` to list available subtasks in error message
            "subtask": Optional[str],
            "tracking_score_threshold": float,
        }
