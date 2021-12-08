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
Human detection and tracking model
"""

from typing import Any, Dict
import os
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.jde_mot import jde_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Node(AbstractNode):
    """Initialise and use the JDE model to detect and track people from
    image frame.

    The JDE model node allows target detection and appearance embedding
    to be learned in a shared model. In addition, the authors We further
    propose a simple and fast association method that works in conjunction
    with the joint model.

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |obj_tags|

        |bbox_labels|

    Configs:
        weights_dir (:obj:`List`):
            Directory pointing to the model weights.
        blob_file (:obj:`str`):
            Mame of file to be downloaded, if weights are not found in
            ``weights_dir``.
        model_files (:obj:`Dict`):
            Dictionary pointing to path of the model weights file and model
            config file.
        iou_threshold (:obj:`float`): **default = 0.5**. |br|
            Threshold value for intersecton over union of detections.
        conf_threshold (:obj:`float`): **default = 0.5**. |br|
            Object confidence score threshold.
        nms_threshold (:obj:`float`): **default = 0.4**. |br|
            Threshold values for non-max suppression.
        min_box_area (:obj:`int`): **default = 200**. |br|
            Minimum value for area of detected bounding box. Calculated
            by width * height.
        track_buffer (:obj:`int`): **default = 30**. |br|
            Threshold to remove track if track is lost for more frames
            than value.

    References:
        Towards Real-Time Multi-Object Tracking:
        https://arxiv.org/abs/1909.12605v2

        Model weights trained by:
        https://github.com/Zhongdao/Towards-Realtime-MOT
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = jde_model.JDE(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run JDE model per frame.

        Args:
            inputs (Dict[str, Any]): Inputs from nodes before JDE.

        Returns:
            Dict[str, Any]: Output dict of model predictions.
        """
        bboxes, obj_tags, labels = self.model.predict(inputs["img"])
        outputs = {
            "bboxes": bboxes,
            "obj_tags": obj_tags,
            "bbox_labels": labels,
        }
        return outputs
