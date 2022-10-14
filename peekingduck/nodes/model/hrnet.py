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

"""🕺 High-Resolution Network: Deep high-resolution representation learning for
human pose estimation. Requires an object detector.
"""


from typing import Any, Dict, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.hrnetv1 import hrnet_model


class Node(AbstractNode):
    """Initializes and uses HRNet model to infer poses from detected bboxes.
    Note that HRNet must be used in conjunction with an object detector applied
    prior.

    The HRNet applied to human pose estimation uses the representation head,
    called HRNetV1.

    The HRNet node is capable of detecting single human figures simultaneously
    per inference, with 17 keypoints estimated for each detected human figure.
    The keypoint indices table can be found
    :ref:`here <whole-body-keypoint-ids>`.

    Inputs:
        |img_data|

        |bboxes_data|

    Outputs:
        |keypoints_data|

        |keypoint_scores_data|

        |keypoint_conns_data|

    Configs:
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        resolution (:obj:`Dict[str, int]`):
            **default = { height: 192, width: 256 }**. |br|
            Resolution of input array to HRNet model.
        score_threshold (:obj:`float`): **[0, 1], default = 0.1**. |br|
            Threshold to determine if detection should be returned

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

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "resolution": Dict[str, int],
            "resolution.height": int,
            "resolution.width": int,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
