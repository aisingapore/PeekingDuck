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

"""
Fast Pose Estimation model
"""

from typing import Dict, Any
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.movenetv1 import movenet_model


class Node(AbstractNode):
    """MoveNet node that initalises a MoveNet model to detect human poses from
    an image.

    The MoveNet node is capable of detecting up to 6 human figures
    simultaneously per inference and for each detected human figure, 17 keypoints
    are estimated. The keypoint indices table can be found :term:`here <keypoint indices>`.

    Inputs:
        |img|

    Outputs:
        |bboxes|


        |keypoints|


        |keypoint_scores|


        |keypoints_conns|


        |bbox_labels|

    Configs:
        model_type (:obj:`str`):  |br|
            **{"                                                                    |br|
            |tab| singlepose_lightning", "singlepose_thunder", "mulitpose_lightning"|br|
            },  default="mulitpose_lightning"**                                     |br|
            Defines the detection model for MoveNet either single or multi pose.
            Lightning is smaller and faster but less accurate than Thunder version.

        weights_dir (:obj:`List`):
            Directory path pointing to folder containing the model weights

        model_weights_dir (:obj: `Dict`):
            Dictionary of filepath to the model weights

        blob_file (:obj:`str`):
            Name of file to be downloaded, if weights are not found in `weights_dir`

        model_files (:obj:`Dict`):
            Dictionary pointing to path of model weights file

        resolution (:obj:`Dict`):  |br|
            Dictionary of resolutions of input array to different MoveNet models.
            Only multipose can use dynamic shape that needs be multiples of 32,
            recommended shape is 256

        bbox_score_threshold (:obj:`float`): **[0,1], default = 0.2** |br|
            Detected bounding box confidence score threshold, only boxes above
            threshold will be kept in the output.

        keypoint_score_threshold (:obj:`float`): **[0,1], default = 0.2** |br|
            Detected keypoints confidence score threshold, only keypoints above
            threshold will be output

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = movenet_model.MoveNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected
        """

        bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"]
        )

        bbox_labels = np.array(["Person"] * len(bboxes))

        return {
            "bboxes": bboxes,
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
            "bbox_labels": bbox_labels,
        }
