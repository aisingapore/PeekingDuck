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

"""🔲 Multi-task Cascaded Convolutional Networks for face detection. Works best
with unmasked faces.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.model.mtcnnv1 import mtcnn_model


class Node(AbstractNode):
    """Initializes and uses the MTCNN model to infer bboxes from an image
    frame.

    The MTCNN node is a single-class model capable of detecting human faces. To
    a certain extent, it is also capable of detecting bounding boxes around
    faces with face masks (e.g. surgical masks).

    Inputs:
        |img_data|

    Outputs:
        |bboxes_data|

        |bbox_scores_data|

        |bbox_labels_data|

    Configs:
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        min_size (:obj:`int`): **default = 40**. |br|
            Minimum height and width of face in pixels to be detected.
        scale_factor (:obj:`float`): **[0, 1], default = 0.709**. |br|
            Scale factor to create the image pyramid. A larger scale factor
            produces more accurate detections at the expense of inference
            speed.
        network_thresholds (:obj:`List[float]`):
            **[0, 1], default = [0.6, 0.7, 0.7]**. |br|
            Threshold values for the Proposal Network (P-Net), Refine Network
            (R-Net) and Output Network (O-Net) in the MTCNN model.

            Calibration is performed at each stage in which bounding boxes with
            confidence scores less than the specified threshold are discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.7**. |br|
            Bounding boxes with confidence scores less than the specified
            threshold in the final output are discarded.

    References:
        Joint Face Detection and Alignment using Multi-task Cascaded
        Convolutional Networks:
        https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf

        Model weights trained by https://github.com/blaueck/tf-mtcnn

    .. versionchanged:: 1.2.0 |br|
        ``mtcnn_min_size`` is renamed to ``min_size``. |br|
        ``mtcnn_factor`` is renamed to ``scale_factor``. |br|
        ``mtcnn_thresholds`` is renamed to ``network_thresholds``. |br|
        ``mtcnn_score`` is renamed to ``score_threshold``.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = mtcnn_model.MTCNNModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reads the image input and returns the bboxes, scores and labels of
        faces detected.

        Args:
            inputs (dict): Dictionary of inputs with key "img".

        Returns:
            outputs (dict): Outputs in dictionary format with keys "bboxes",
            "bbox_scores", and "bbox_labels".
        """
        bboxes, bbox_scores, _ = self.model.predict(inputs["img"])
        bbox_labels = np.array(["face"] * len(bboxes))
        bboxes = np.clip(bboxes, 0, 1)

        return {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "bbox_scores": bbox_scores,
        }

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {
            "min_size": int,
            "network_thresholds": List[float],
            "scale_factor": float,
            "score_threshold": float,
            "weights_parent_dir": Optional[str],
        }
