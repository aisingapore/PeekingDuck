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

"""High performance anchor-free YOLO object detection model"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.yoloxv1 import yolox_model
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses YOLOX to infer from an image frame.

    The YOLOX node is capable detecting objects from 80 categories. The table
    of object categories can be found
    :ref:`here <general-object-detection-ids>`. The ``"yolox-tiny"`` model is
    used by default and can be changed to one of ``("yolox-tiny", "yolox-s",
    "yolox-m", "yolox-l")``.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

    Configs:
        model_type (:obj:`str`): **{"yolox-tiny", "yolox-s", "yolox-m",
            "yolox-l"}, default="yolox-tiny"**. |br|
            Defines the type of YOLOX model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        input_size (:obj:`int`): **default=416**. |br|
            Input image resolution of the YOLOX model.
        detect_ids (:obj:`List[int]`): **default=[0]**. |br|
            List of object category IDs to be detected. To detect all classes,
            refer to the :ref:`tech note <general-object-detection-ids>`.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.45**. |br|
            Overlapping bounding boxes with Intersection over Union (IoU) above
            the threshold will be discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.25**. |br|
            Bounding boxes with confidence score (product of objectness score
            and classification score) below the threshold will be discarded.
        agnostic_nms (:obj:`bool`): **default = True**. |br|
            Flag to determine if class agnostic NMS (``torchvision.ops.nms``)
            or class aware NMS (``torchvision.ops.batched_nms``) should be
            used.
        half (:obj:`bool`): **default = False**. |br|
            Flag to determine if half-precision floating-point should be used
            for inference.
        fuse (:obj:`bool`): **default = False**. |br|
            Flag to determine if the convolution and batch normalization layers
            should be fused for inference.

    References:
        YOLOX: Exceeding YOLO Series in 2021:
        https://arxiv.org/abs/2107.08430

        Inference code and model weights:
        https://github.com/Megvii-BaseDetection/YOLOX
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = yolox_model.YOLOXModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes of the detect
        objects.

        The classes of objects to be detected can be specified through the
        `detect_ids` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs
