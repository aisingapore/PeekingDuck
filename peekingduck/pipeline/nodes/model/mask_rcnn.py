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


from typing import Any, Dict

from peekingduck.pipeline.nodes.model.mask_rcnnv1 import mask_rcnn_model
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mask_rcnn_model.MaskRCNNModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads `img` from `inputs` and return the bboxes and masks of the detect
        objects.

        The classes of objects to be detected can be specified through the
        `detect_ids` configuration option.

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
