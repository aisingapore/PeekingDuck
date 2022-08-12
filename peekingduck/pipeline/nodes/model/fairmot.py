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

"""🎯 Human detection and tracking model that balances the importance between
detection and re-ID tasks.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.fairmotv1 import fairmot_model


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses FairMOT tracking model to detect and track people
    from the supplied image frame.

    FairMOT is based on the anchor-free object detector CenterNet with
    modifications to balance the importance between detection and
    re-identification tasks in an object tracker.

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
        score_threshold (:obj:`float`): **default = 0.5**. |br|
            Object confidence score threshold.
        K (:obj:`int`): **default = 500**. |br|
            Maximum number of objects output during the object detection stage.
        min_box_area (:obj:`int`): **default = 100**. |br|
            Minimum value for area of detected bounding box. Calculated by
            width * height.
        track_buffer (:obj:`int`): **default = 30**. |br|
            Threshold to remove track if track is lost for more frames
            than value.
        input_size (:obj:`List[int]`): **default = [864, 480]**. |br|
            Size (width, height) of the input image to the model. Raw
            video/image frames will be resized to the ``input_size`` before
            they are fed to the model.

    References:
        FairMOT: On the Fairness of Detection and Re-Identification in Multiple
        Object Tracking
        https://arxiv.org/abs/2004.01888

        Model weights trained by:
        https://github.com/ifzhang/FairMOT
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._frame_rate = 30.0

        self.model = fairmot_model.FairMOTModel(self.config, self._frame_rate)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks objects from image.

        Specifically for use with MOT evaluation, will attempt to get optional
        input `mot_metadata` and recreate `FairMOTModel` with the appropriate
        frame rate when necessary.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img". When running
                under MOT evaluation, contains "mot_metadata" key as well.

        Returns:
            (Dict[str, Any]): Dictionary containing:
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

        outputs = {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "bbox_scores": bbox_scores,
            "obj_attrs": {"ids": track_ids},
        }
        return outputs

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "input_size": List[int],
            "K": int,
            "min_box_area": int,
            "score_threshold": float,
            "track_buffer": int,
            "weights_parent_dir": Optional[str],
        }

    def _reset_model(self) -> None:
        """Creates a new instance of the FairMOT model with the frame rate
        supplied by `mot_metadata`.
        """
        self.logger.info(
            f"Creating new model with frame rate: {self._frame_rate:.2f}..."
        )
        self.model = fairmot_model.FairMOTModel(self.config, self._frame_rate)
