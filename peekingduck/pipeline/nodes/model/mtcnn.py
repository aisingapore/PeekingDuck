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
Fast face detection model that works best with unmasked faces.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.model.mtcnnv1 import mtcnn_model
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Initialises and use the MTCNN model to infer bboxes from image frame.

    The MTCNN node is a single-class model capable of detecting human faces. To
    a certain extent, it is also capable of detecting bounding boxes around
    faces with face masks (e.g. surgical masks).

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_scores|

        |bbox_labels|

    Configs:
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by replacing
            ``null`` with an absolute path to the desired directory.
        mtcnn_min_size (:obj:`int`): **default = 40**. |br|
            Minimum height and width of face in pixels to be detected.
        mtcnn_factor (:obj:`float`): **[0, 1], default = 0.709**. |br|
            Scale factor to create the image pyramid. A larger scale factor
            produces more accurate detections at the expense of inference
            speed.
        mtcnn_thresholds (:obj:`List`):
            **[0, 1], default = [0.6, 0.7, 0.7]**. |br|
            Threshold values for the Proposal Network (P-Net), Refine Network
            (R-Net) and Output Network (O-Net) in the MTCNN model.

            Calibration is performed at each stage in which bounding boxes with
            confidence scores less than the specified threshold are discarded.
        mtcnn_score (:obj:`float`): **[0, 1], default = 0.7**. |br|
            Bounding boxes with confidence scores less than the specified
            threshold in the final output are discarded.

    References:
        Joint Face Detection and Alignment using Multi-task Cascaded
        Convolutional Networks:
        https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf

        Model weights trained by https://github.com/blaueck/tf-mtcnn
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mtcnn_model.MtcnnModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the bboxes, scores and labels of
        faces detected.

        Args:
            inputs (dict): Dictionary of inputs with key "img".

        Returns:
            outputs (dict): Outputs in dictionary format with keys "bboxes",
            "bbox_scores", and "bbox_labels".
        """
        bboxes, scores, _, classes = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_scores": scores, "bbox_labels": classes}
        return outputs
