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
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.posenetv1 import posenet_model


class Node(AbstractNode):
    """PoseNet node that initalises a PoseNet model to detect human poses from
    an image.

    The PoseNet node is capable of detecting multiple human figures
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
        model_type (:obj:`str`): **{"resnet", "50", "75", "100"}, default="resnet"**

            Defines the backbone model for PoseNet.

        weights_dir (:obj:`List`):
            list of directories pointing to model weights

        blob_file (:obj:`str`):
            name of file to be downloaded, if weights are not found in `weights_dir`

        model_files (:obj:`Dict`):
            dictionary pointing to path of model weights file

        resolution (:obj:`Dict`): **default = { height: 225, width: 225 }**

            resolution of input array to PoseNet model

        max_pose_detection (:obj:`int`): **default = 10**

            maximum number of poses to be detected

        score_threshold (:obj:`float`): **[0,1], default = 0.4**

            threshold to determine if detection should be returned


    References:

    PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up,
        Part-Based, Geometric Embedding Model:
        https://arxiv.org/abs/1803.08225

    Code adapted from https://github.com/rwightman/posenet-python


    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = posenet_model.PoseNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the image input and returns the bboxes
        of the specified objects chosen to be detected
        """
        bboxes, keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"])
        bbox_labels = ["Person"] * len(bboxes)
        outputs = {"bboxes": bboxes,
                   "keypoints": keypoints,
                   "keypoint_scores": keypoint_scores,
                   "keypoint_conns": keypoint_conns,
                   "bbox_labels": bbox_labels}
        return outputs
