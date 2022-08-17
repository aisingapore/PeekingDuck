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

"""🎭 Instance segmentation model for generating high-quality masks."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.mask_rcnnv1 import mask_rcnn_model
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Initializes and uses Mask R-CNN to infer from an image frame.

    The Mask-RCNN node is capable detecting objects and their respective masks
    from 80 categories. The table of object categories can be found
    :ref:`here <general-instance-segmentation-ids>`. The ``"r50-fpn"`` backbone is
    used by default, and the ``"r101-fpn"`` for the ResNet 101 backbone variant can also
    be chosen.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

        |masks_data|

    Configs:
        model_type (:obj:`str`): **{"r50-fpn", "r101-fpn"}, default = "r50-fpn"**. |br|
            Defines the type of backbones to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        min_size (:obj:`int`): **default = 800**. |br|
            Minimum size of the image to be rescaled before feeding it to the
            backbone.
        max_size (:obj:`int`): **default = 1333**. |br|
            Maximum size of the image to be rescaled before feeding it to the
            backbone.
        detect (:obj:`List[Union[int, string]]`): **default = [0]**. |br|
            List of object class names or IDs to be detected. To detect all classes,
            refer to the :ref:`tech note <general-instance-segmentation-ids>`.
        max_num_detections: (:obj:`int`): **default = 100**. |br|
            Maximum number of detections per image, for all classes.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Overlapping bounding boxes with Intersection over Union (IoU) above
            the threshold will be discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Bounding boxes with classification score below the threshold will be discarded.
        mask_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            The confidence threshold for binarizing the masks' pixel values; determines whether an
            object is detected at a particular pixel.

    References:
        Mask R-CNN: A conceptually simple, flexible, and general framework for object
        instance segmentation.:
        https://arxiv.org/abs/1703.06870

        Inference code adapted from:
        https://pytorch.org/vision/0.11/_modules/torchvision/models/detection/mask_rcnn.html

        The weights for Mask-RCNN Model with ResNet50 FPN backbone were adapted from:
        https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mask_rcnn_model.MaskRCNNModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes and masks of the detect
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
