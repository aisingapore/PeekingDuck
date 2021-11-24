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
Fast face detection model that can distinguish between masked and unmasked
faces.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.yolov4_face import yolo_face_model
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initialises and use the YOLO face detection model to infer bboxes from
    image frame.

    The YOLO face model is a two class model capable of differentiating human
    faces with and without face masks.

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_labels|

        |bbox_scores|

    Configs:
        model_type (:obj:`str`): **{"v4", "v4tiny"}, default="v4tiny"**. |br|
            Defines the type of YOLO model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by replacing
            ``null`` with an absolute path to the desired directory.
        detect_ids (:obj:`List`): **default = [0, 1]**. |br|
            List of object class IDs to be detected where `no_mask` is ``0``
            and `mask` is ``1``.
        max_output_size_per_class (:obj:`int`): **default = 50**. |br|
            Maximum number of detected instances for each class in an image.
        max_total_size (:obj:`int`): **default = 50**. |br|
            Maximum total number of detected instances in an image.
        yolo_iou_threshold (:obj:`float`): **[0, 1], default = 0.1**. |br|
            Overlapping bounding boxes above the specified IoU (Intersection
            over Union) threshold are discarded.
        yolo_score_threshold (:obj:`float`): **[0, 1], default = 0.7**. |br|
            Bounding box with confidence score less than the specified
            confidence score threshold is discarded.

    References:
        YOLOv4: Optimal Speed and Accuracy of Object Detection:
        https://arxiv.org/pdf/2004.10934v1.pdf

        Model weights trained using pretrained weights from Darknet:
        https://github.com/AlexeyAB/darknet
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = yolo_face_model.Yolov4(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {
            "bboxes": bboxes,
            "bbox_labels": labels,
            "bbox_scores": scores,
        }

        return outputs
