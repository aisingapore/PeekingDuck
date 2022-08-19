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

"""🕺 Fast Pose Estimation model."""

from typing import Any, Dict, Optional, Union

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.posenetv1 import posenet_model


class Node(AbstractNode):
    """Initializes a PoseNet model to detect human poses from an image.

    The PoseNet node is capable of detecting multiple human figures
    simultaneously per inference and for each detected human figure, 17
    keypoints are estimated. The keypoint indices table can be found
    :ref:`here <whole-body-keypoint-ids>`.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

        |bbox_labels_data|

    Configs:
        model_type (:obj:`Union[str, int]`):
            **{"resnet", 50, 75, 100}, default="resnet"**. |br|
            Defines the backbone model for PoseNet.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        resolution (:obj:`Dict`):
            **default = { height: 225, width: 225 }**. |br|
            Resolution of input array to PoseNet model.
        max_pose_detection (:obj:`int`): **default = 10**. |br|
            Maximum number of poses to be detected.
        score_threshold (:obj:`float`): **[0, 1], default = 0.4**. |br|
            Detected keypoints confidence score threshold, only keypoints above
            threshold will be kept in output.

    References:
        PersonLab: Person Pose Estimation and Instance Segmentation with a
        Bottom-Up, Part-Based, Geometric Embedding Model:
        https://arxiv.org/abs/1803.08225

        Code adapted from https://github.com/rwightman/posenet-python
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = posenet_model.PoseNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the bboxes of the specified
        objects chosen to be detected.
        """
        bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"]
        )
        bbox_labels = np.array(["person"] * len(bboxes))
        bboxes = np.clip(bboxes, 0, 1)

        outputs = {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
        }
        return outputs

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "max_pose_detection": int,
            "model_type": Union[str, int],
            "resolution": Dict[str, int],
            "resolution.height": int,
            "resolution.width": int,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
