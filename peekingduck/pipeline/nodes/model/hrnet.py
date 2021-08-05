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
Slower, accurate Pose Estimation model. Requires a object detector
"""


from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.hrnetv1 import hrnet_model


class Node(AbstractNode):
    """HRNet node class that initialises and use hrnet model to infer poses
    from detected bboxes. Note that HRNet must be used in conjunction with
    a object detector applied prior.

    The HRNet applied to human pose estimation uses the representation head,
    called HRNetV1.

    The HRNet node is capable of detecting single human figures
    simultaneously per inference and for each detected human figure,
    17 keypoints are estimated. The keypoint indices table can be found
    :term:`here <keypoint indices>`.

    Inputs:
        |img|


        |bboxes|

    Outputs:
        |keypoints|


        |keypoint_scores|


        |keypoints_conns|

    Configs:
        weights_dir (:obj:`List`):
            list of directories pointing to model weights

        blob_file (:obj:`str`):
            name of file to be downloaded, if weights are not found in `weights_dir`

        model_files (:obj:`Dict`):
            dictionary pointing to path of model weights file

        resolution (:obj:`Dict`): **default = { height: 192, width: 256 }**

            resolution of input array to HRNet model

        score_threshold (:obj:`float`): **[0,1], default = 0.1**

            threshold to determine if detection should be returned

        model_nodes (:obj:`Dict`):  **default = { inputs: [x:0], outputs: [Identity:0] }**

            names of input and output nodes from model graph for prediction


    References:
    Deep High-Resolution Representation Learning for Visual Recognition:
    https://arxiv.org/abs/1908.07919
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = hrnet_model.HRNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """function that reads the bbox input and returns the poses
        and pose bbox of the specified objects chosen to be detected"""
        keypoints, keypoint_scores, keypoint_conns = self.model.predict(
            inputs["img"], inputs["bboxes"])

        outputs = {"keypoints": keypoints,
                   "keypoint_scores": keypoint_scores,
                   "keypoint_conns": keypoint_conns}
        return outputs
