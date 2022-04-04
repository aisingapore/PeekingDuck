# Copyright 2022 AI Singapore
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

"""🎯 Joint Detection and Embedding model for human detection and tracking."""

from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.model.jdev1 import jde_model
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Initializes and uses JDE tracking model to detect and track people from
    the supplied image frame.

    JDE is a fast and high-performance multiple-object tracker that learns the
    object detection task and appearance embedding task simultaneously in a
    shared neural network.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

        |obj_attrs_data|
        :mod:`model.fairmot` produces the ``ids`` attribute which contains the
        tracking IDs of the detections.

    Configs:
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        iou_threshold (:obj:`float`): **default = 0.5**. |br|
            Threshold value for Intersecton-over-Union of detections.
        nms_threshold (:obj:`float`): **default = 0.4**. |br|
            Threshold values for non-max suppression.
        score_threshold (:obj:`float`): **default = 0.5**. |br|
            Object confidence score threshold.
        min_box_area (:obj:`int`): **default = 200**. |br|
            Minimum value for area of detected bounding box. Calculated by
            :math:`width \\times height`.
        track_buffer (:obj:`int`): **default = 30**. |br|
            Threshold to remove track if track is lost for more frames than
            value.

    References:
        Towards Real-Time Multi-Object Tracking:
        https://arxiv.org/abs/1909.12605v2

        Model weights trained by:
        https://github.com/Zhongdao/Towards-Realtime-MOT
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._frame_rate = 30.0

        self.model = jde_model.JDEModel(self.config, self._frame_rate)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks objects from image.

        Specifically for use with MOT evaluation, will attempt to get optional
        input `mot_metadata` and recreate `JDEModel` with the appropriate
        frame rate when necessary.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img". When running
                under MOT evaluation, contains "mot_metadata" key as well.

        Returns:
            outputs (dict): Dictionary containing:
            - bboxes (List[np.ndarray]): Bounding boxes for tracked targets.
            - bbox_labels (np.ndarray): Bounding box labels, hard coded as
                "person".
            - bbox_scores (List[float]): Detection confidence scores.
            - obj_attrs (Dict[str, List[int]]): Tracking IDs, specifically for use
                with `mot_evaluator`.
        """
        metadata = inputs.get(
            "mot_metadata", {"frame_rate": self._frame_rate, "reset_model": False}
        )
        frame_rate = metadata["frame_rate"]
        reset_model = metadata["reset_model"]

        if frame_rate != self._frame_rate or reset_model:
            self._frame_rate = frame_rate
            self._reset_model()

        bboxes, bbox_scores, track_ids = self.model.predict(inputs["img"])
        bbox_labels = np.array(["person"] * len(bboxes))
        bboxes = np.clip(bboxes, 0, 1)

        return {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "bbox_scores": bbox_scores,
            "obj_attrs": {"ids": track_ids},
        }

    def _reset_model(self) -> None:
        """Creates a new instance of the JDE model with the frame rate
        supplied by `mot_metadata`.
        """
        self.logger.info(
            f"Creating new model with frame rate: {self._frame_rate:.2f}..."
        )
        self.model = jde_model.JDEModel(self.config, self._frame_rate)
