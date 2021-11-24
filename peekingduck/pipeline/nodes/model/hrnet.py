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
Slower but more accurate Pose Estimation model. Requires an object detector.
"""


from typing import Any, Dict

from peekingduck.pipeline.nodes.model.hrnetv1 import hrnet_model
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Initialises and use HRNet model to infer poses from detected bboxes.
    Note that HRNet must be used in conjunction with an object detector applied
    prior.

    The HRNet applied to human pose estimation uses the representation head,
    called HRNetV1.

    The HRNet node is capable of detecting single human figures simultaneously
    per inference and for each detected human figure, 17 keypoints are
    estimated. The keypoint indices table can be found
    :ref:`here <whole-body-keypoint-ids>`.

    Inputs:
        |img|

        |bboxes|

    Outputs:
        |keypoints|

        |keypoint_scores|

        |keypoint_conns|

    Configs:
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by replacing
            ``null`` with an absolute path to the desired directory.
        resolution (:obj:`Dict`):
            **default = { height: 192, width: 256 }**. |br|
            Resolution of input array to HRNet model.
        score_threshold (:obj:`float`): **[0, 1], default = 0.1**. |br|
            Threshold to determine if detection should be returned
        model_nodes (:obj:`Dict`):
            **default = { inputs: [x:0], outputs: [Identity:0] }** |br|
            Names of input and output nodes from model graph for prediction.

    References:
        Deep High-Resolution Representation Learning for Visual Recognition:
        https://arxiv.org/abs/1908.07919
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = hrnet_model.HRNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the bbox input and returns the poses and pose bbox of the
        specified objects chosen to be detected.
        """
        keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"], inputs["bboxes"]
        )

        outputs = {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "keypoint_conns": keypoint_conns,
        }
        return outputs
