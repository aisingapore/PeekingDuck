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
Person Re-Identification Model using OSNet.
"""

from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.model.osnetv1 import osnet_model


class Node(AbstractNode):
    """Person re-identification node class that initialises and uses
    OSNet model to infer matching bboxes of a queried image from frames.

    The reid node is capable of detecting a matching object based on the
    query. OSNet has been trained on persons datasets.

    Inputs:
        |img|

        |bboxes|

        |bbox_labels|

    Outputs:
        |bboxes|

        |obj_tags|

    Configs:
        device (:obj:`str`): **{"cpu", "cuda"}, default="cpu"**  |br|
            Defines the device/ processor to be used.
        model_type (:obj:`str`): **{"osnet", "osnet_ain"}, default="osnet"**  |br|
            Defines the type of model that is to be used.
        query_root_dir (:obj:`str`):
            Path pointing to the root folder where cropped images to be
            queried are stored. If path to image is `[root_path]/reid/person1/img1.jpg`,
            `query_root_dir = [root_path]/reid`
        multi_threshold (:obj:`float`): **default=0.3**  |br|
            Threshold for cosine distance matching for multiple queries' features.

    References:
        Omni-Scale Feature Learning for Person Re-Identification:
            https://arxiv.org/abs/1905.00953

        Learning Generalisable Omni-Scale Representations for Person Re-Identification:
            https://arxiv.org/abs/1910.06827

        Model weights and inference code adapted from:
            https://github.com/KaiyangZhou/deep-person-reid
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = osnet_model.OSNetModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Function that reads the image input and returns the matching
        bbox of the specified query object to be detected.

        Args:
            inputs (Dict): Dict of inputs with keys "img", "bboxes".

        Returns:
            outputs (Dict): Dict output of matching bbox with object tag
                label. Label comprises of identifier of the person of interest.
        """
        matching_info = self.model.predict(inputs["img"], inputs["bboxes"])
        names = list(matching_info.keys())
        matching_bbox = list(matching_info.values())

        outputs = {"bboxes": matching_bbox, "obj_tags": names}

        return outputs
