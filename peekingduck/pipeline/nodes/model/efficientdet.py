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
Slower but more accurate object detection model.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.efficientdet_d04 import efficientdet_model
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Initialises an EfficientDet model to detect bounding boxes from an image.

    The EfficientDet node is capable of detecting objects from 80 categories.
    The table of categories can be found
    :ref:`here <general-object-detection-ids>`.

    EfficientDet node has five levels of compound coefficient (0 - 5). A higher
    compound coefficient will scale up all dimensions of the backbone network
    width, depth, and input resolution, which results in better performance but
    slower inference time. The default compound coefficient is 0 and can be
    changed to other values.

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_labels|

        |bbox_scores|

    Configs:
        model_type (:obj:`int`): **{0, 1, 2, 3, 4}, default = 0**. |br|
            Defines the compound coefficient for EfficientDet.
        score_threshold (:obj:`float`): **[0, 1], default = 0.3**.
            Threshold to determine if detection should be returned.
        detect_ids (:obj:`List[int]`): **default = [0]**. |br|
            List of object class IDs to be detected.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by replacing
            ``null`` with an absolute path to the desired directory.

    References:
        EfficientDet: Scalable and Efficient Object Detection:
        https://arxiv.org/abs/1911.09070

        Code adapted from https://github.com/xuannianz/EfficientDet.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = efficientdet_model.EfficientDetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Takes an image as input and returns bboxes of objects specified
        in config.
        """
        # Currently prototyped to return just the bounding boxes
        # without the scores
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
