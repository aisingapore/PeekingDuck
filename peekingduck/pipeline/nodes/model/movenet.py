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

from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.movenetv1 import movenet_model


class Node(AbstractNode):
    """MoveNet node that initializes a MoveNet model to detect human poses from
    an image.

    The MoveNet node is capable of detecting up to 6 human figures for
    multipose lightning and single person for singlepose lightning/thunder. If
    there are more than 6 persons in the image, multipose lightning will only
    detect 6. This also applies to singlepose models, where only 1 person will
    be detected in a multi persons image, do take note that detection
    performance will suffer when using singlepose models on multi persons
    images. 17 keypoints are estimated and the keypoint indices table can be
    found :ref:`here <whole-body-keypoint-ids>`.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

        |bbox_labels_data|

    Configs:
        model_type (:obj:`str`):
            **{"
            singlepose_lightning", "singlepose_thunder", "multipose_lightning"
            },  default="multipose_lightning"** |br|
            Defines the detection model for MoveNet either single or multi pose.
            Lightning is smaller and faster but less accurate than Thunder
            version.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        resolution (:obj:`Dict`): **default = { height: 256, width: 256 }** |br|
            Dictionary of resolutions of input array to different MoveNet models.
            Only multipose allows dynamic shape in multiples of 32 (recommended
            256). Default will be the resolution for multipose lightning model.
        bbox_score_threshold (:obj:`float`): **[0,1], default = 0.2** |br|
            Detected bounding box confidence score threshold, only boxes above
            threshold will be kept in the output.
        keypoint_score_threshold (:obj:`float`): **[0,1], default = 0.3** |br|
            Detected keypoints confidence score threshold, only keypoints above
            threshold will be kept in output.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = movenet_model.MoveNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Function that reads the image input and returns the bboxes,
        keypoints, keypoints confidence scores, keypoint connections and
        bounding box labels of the persons detected.

        Args:
            inputs (Dict[str, Any]): Dictionary of inputs with key "img".

        Returns:
            (Dict[str, Any]): bbox output in dictionary format with keys
            "bboxes", "keypoints", "keypoint_scores", "keypoint_conns", and
            "bbox_labels".
        """
        image = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
        bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(image)
        bbox_labels = np.array(["person"] * len(bboxes))
        bboxes = np.clip(bboxes, 0, 1)

        return {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "keypoints": keypoints,
            "keypoint_conns": keypoint_conns,
            "keypoint_scores": keypoint_scores,
        }
