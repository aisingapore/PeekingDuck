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

"""🎭 Instance segmentation model for real-time inference"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.yolact_edgev1 import yolact_edge_model
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Initializes and uses YolactEdge to infer from an image frame

    The YolactEdge node is capable of detecting objects from 80 categories.
    The table of object categories can be found
    :ref:`here <general-object-detection-ids>`.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

        |masks_data|

    Configs:
        model_type (:obj:`str`): (:obj:`str`): **{"r101-fpn", "r50-fpn",
            "mobilenetv2"}, default="r50-fpn"**. |br|
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        input_size (:obj:`int`): **default = 550**. |br|
            Input image resolution of the YolactEdge model.
        detect (:obj:`List[Union[int, string]]`): **default=[0]**. |br|
            List of object class names or IDs to be detected. To detect all classes,
            refer to the :ref:`tech note <general-object-detection-ids>`.
        max_num_detections: (:obj:`int`): **default=100**. |br|
            Maximum number of detections per image, for all classes.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Overlapping bounding boxes with Intersection over Union (IoU) above
            the threshold will be discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.2**. |br|
            Bounding boxes with confidence score (product of objectness score
            and classification score) below the threshold will be discarded.


    References:
        YolactEdge: Real-time Instance Segmentation on the Edge
        https://arxiv.org/abs/2012.12259

        Inference code and model weights:
        https://github.com/haotian-liu/yolact_edge
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = yolact_edge_model.YolactEdgeModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes and masks of the detected
        objects.

        The classes of objects to be detected can be specified through the
        `detect` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                `bbox_scores` and `masks`.
        """
        bboxes, labels, scores, masks = self.model.predict(inputs["img"])

        outputs = {
            "bboxes": bboxes,
            "bbox_labels": labels,
            "bbox_scores": scores,
            "masks": masks,
        }

        return outputs
