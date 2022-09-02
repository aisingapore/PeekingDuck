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
from peekingduck.pipeline.nodes.model.huggingfacev1 import huggingface_model


class Node(AbstractNode):
    """Initializes and uses Hugging Face Hub models to infer from an image
    frame.

    _TODO: <Explanations about detect ID>

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
            Defines the type of YOLOX model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default=null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        detect (:obj:`List[Union[int, str]]`): **default=[0]**. |br|
            List of object class names or IDs to be detected. To detect all
            classes, refer to the :ref:`tech note <general-object-detection-ids>`.

    References:
        Hugging Face Hub
        https://huggingface.co/docs/hub/index
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = huggingface_model.ObjectDetectionModel(self.config)

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
