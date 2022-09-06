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

from typing import Any, Dict

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.huggingface_hubv1 import huggingface_hub_model


class Node(AbstractNode):
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

    Configs:
        task (:obj:`str`): **{"object_detection"},
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
        iou_threshold (:obj:`float`): **[0, 1], default = 0.6**. |br|
            Overlapping bounding boxes with Intersection over Union (IoU) above
            the threshold will be discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.5**. |br|
            Bounding boxes with confidence score below the threshold will be
            discarded.
        agnostic_nms (:obj:`bool`): **default = True**. |br|
            Flag to determine if class-agnostic NMS (``torchvision.ops.nms``)
            or class-aware NMS (``torchvision.ops.batched_nms``) should be
            used.

    References:
        Hugging Face Hub
        https://huggingface.co/docs/hub/index
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = huggingface_hub_model.ObjectDetectionModel(self.config)
        self.config["detect"] = self.model.detect_ids

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
        bboxes, labels, scores = self.model.predict(image)
        bboxes = np.clip(bboxes, 0, 1)

        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs
