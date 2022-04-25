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

"""🔲 License Plate Detection model."""

from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.model.yolov4_license_plate import (
    yolo_license_plate_model,
)
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses YOLO model to infer bboxes from image frame.

    This customized YOLO node is capable of detecting objects from a single
    class (License Plate). It uses YOLOv4 by default and can be changed to
    use YOLOv4-tiny if FPS is critical over accuracy.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:
        model_type (:obj:`str`): **{"v4", "v4tiny"}, default="v4"**. |br|
            Defines the type of YOLO model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.3**. |br|
            Overlapping bounding boxes above the specified IoU (Intersection
            over Union) threshold are discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.1**. |br|
            Bounding box with confidence score less than the specified
            confidence score threshold is discarded.

    References:
        YOLOv4: Optimal Speed and Accuracy of Object Detection:
        https://arxiv.org/pdf/2004.10934v1.pdf

        Model weights trained using pretrained weights from Darknet:
        https://github.com/AlexeyAB/darknet

    .. versionchanged:: 1.2.0
        ``yolo_iou_threshold`` is renamed to ``iou_threshold``.

    .. versionchanged:: 1.2.0
        ``yolo_score_threshold`` is renamed to ``score_threshold``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = yolo_license_plate_model.YOLOLicensePlateModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the bboxes of the specified
        objects chosen to be detected.

        Args:
            inputs (dict): Dictionary of inputs with key "img".
        Returns:
            outputs (dict): bbox output in dictionary format with keys
            "bboxes", "bbox_labels", and "bbox_scores".
        """
        image = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = self.model.predict(image)
        bboxes = np.clip(bboxes, 0, 1)

        outputs = {
            "bboxes": bboxes,
            "bbox_labels": labels,
            "bbox_scores": scores,
        }

        return outputs
