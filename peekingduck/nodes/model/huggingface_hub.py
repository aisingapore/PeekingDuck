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

"""Hugging Face Hub models for computer vision tasks."""

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.base import ThresholdCheckerMixin
from peekingduck.pipeline.nodes.model.huggingface_hubv1 import models


class Node(ThresholdCheckerMixin, AbstractNode):
    """Initializes and uses Hugging Face Hub models to infer from an image
    frame.

    The models from the Hugging Face Hub are trained on a variety of datasets
    and can predict a variety of labels. Please refer to the documentation for
    ``model_type`` and ``detect`` below for details on how to configure the
    node.

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_labels_data|

        |bbox_scores_data|

        |masks_data| (Only available when ``task = instance_segmentation``)

    Configs:
        task (:obj:`str`): **{"instance_segmentation", "object_detection"},
            default="object_detection"** |br|
            Defines the computer vision task of the model.
        model_type (:obj:`str`): **Refer to CLI command, default=null**. |br|
            Defines the type of Hugging Face Hub model to be used. Use the CLI
            command ``peekingduck model-hub huggingface models --task '<task>'``
            to obtain a list of available models.
        weights_parent_dir (:obj:`Optional[str]`): **default=null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        detect (:obj:`List[Union[int, str]]`): **default=[0]**. |br|
            List of object class names or IDs to be detected. Please refer to
            the :ref:`tech note <general-object-detection-ids>` for the class
            name/ID mapping and instructions on detects all classes.
        score_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Bounding boxes with confidence score below the threshold will be
            discarded.
        mask_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            The confidence threshold for binarizing the masks' pixel values;
            determines whether an object is detected at a particular pixel.
            Used when ``task = instance_segmentation``.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.6**. |br|
            Overlapping bounding boxes with Intersection over Union (IoU) above
            the threshold will be discarded. Used when
            ``task = object_detection``.
        agnostic_nms (:obj:`bool`): **default = True**. |br|
            Flag to determine if class-agnostic NMS (``torchvision.ops.nms``)
            or class-aware NMS (``torchvision.ops.batched_nms``) should be
            used. Used when ``task = object_detection``.

    References:
        Hugging Face Hub
        https://huggingface.co/docs/hub/index
    """

    model_constructor = {
        "instance_segmentation": models.InstanceSegmentationModel,
        "object_detection": models.ObjectDetectionModel,
    }

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.check_valid_choice("task", {"instance_segmentation", "object_detection"})
        self.model = self.model_constructor[self.config["task"]](self.config)
        self.model.post_init()
        self.config["detect"] = self.model.detect_ids
        self._finalize_output_keys()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` perform prediction on it.

        The classes of objects to be detected can be specified through the
        `detect` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        image = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)

        # bboxes, bbox_labels, bbox_scores for object_detection
        # bboxes, bbox_labels, bbox_scores, masks for instance_segmentation
        results = self.model.predict(image)
        bboxes = np.clip(results[0], 0, 1)

        if self.config["task"] == "object_detection":
            return {
                "bboxes": bboxes,
                "bbox_labels": results[1],
                "bbox_scores": results[2],
            }

        return {
            "bboxes": bboxes,
            "bbox_labels": results[1],
            "bbox_scores": results[2],
            "masks": results[3],
        }

    def _finalize_output_keys(self) -> None:
        """Updates output keys based on the selected ``task``."""
        self.config["output"] = self.config["output"][self.task]
        self.output = self.config["output"]

    def _get_config_types(self) -> Dict[str, Any]:
        return {
            "agnostic_nms": bool,
            "detect": List[Union[int, str]],
            "iou_threshold": float,
            "mask_threshold": float,
            # Allow `None` to trigger listing available model types in error message
            "model_type": Optional[str],
            "score_threshold": float,
            "task": str,
            "weights_parent_dir": Optional[str],
        }
