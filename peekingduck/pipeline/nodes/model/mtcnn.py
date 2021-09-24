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
MTCNN Detection model
"""

from typing import Dict, Any

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.mtcnnv1 import mtcnn_model


class Node(AbstractNode):
    """MTCNN node class that initialises and use the MTCNN model to infer bboxes
    from image frame.

    The MTCNN node is a single-class model capable of detecting human faces. To
    a certain extent, it is also capable of detecting humans wearing face masks
    (e.g. surgical masks).

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_scores|

        |bbox_labels|

    Configs:
        weights_dir (:obj:`List`):
            directory pointing to the model weights.

        blob_file (:obj:`str`):
            name of file to be downloaded, if weights are not found in `weights_dir`.

        graph_files (:obj:`Dict`):
            dictionary pointing to path of the model weights file.

        mtcnn_min_size (:obj:`int`): **default = 40 **

            minimum size of face to be detected.

        mtcnn_factor (:obj:`float`): **default = 0.709 **

            scale factor to build the image pyramid.

        mtcnn_thresholds (:obj:`List`): **[0,1], default = [0.6, 0.7, 0.7]**

            threshold values for each of the networks in the MTCNN model.

        mtcnn_score (:obj:`float`): **[0,1], default = 0.7**

            bounding box with confidence score less than the specified
            confidence score threshold is discarded.

    References:

    Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks:
        https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf

    Model weights trained by https://github.com/blaueck/tf-mtcnn
    """
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mtcnn_model.MtcnnModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Function that reads the image input and returns the bboxes, scores
        and labels of faces detected

        Args:
            inputs (Dict): Dictionary of inputs with key "img"

        Returns:
            outputs (Dict): Outputs in dictionary format with keys "bboxes",
            "bbox_scores" and "bbox_labels"
        """
        bboxes, scores, _, classes = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_scores": scores, "bbox_labels": classes}
        return outputs
