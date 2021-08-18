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

"""

import logging
from typing import Any, Dict, List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.dabble.coco_utils.pack_detections import PackDetections
from peekingduck.pipeline.nodes.dabble.coco_utils.pack_keypoints import PackKeypoints

BOUNDING_BOX = ["bboxes", "bbox_labels", "bbox_scores"]
KEYPOINTS = ["keypoints", "keypoint_scores"]


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        
        self.logger = logging.getLogger(__name__)

        self.evaluation_type = config['type']
        if self.evaluation_type == "detection":
            config['input'] = config['input'] + BOUNDING_BOX
            self.packer = PackDetections()
            self.coco_instance = COCO(config['instances_dir'])
        elif self.evaluation_type == "keypoints":
            config['input'] = config['input'] + KEYPOINTS
            self.packer = PackKeypoints()
            self.coco_instance = COCO(config['keypoints_dir'])

        super().__init__(config, node_path=__name__, **kwargs)

        self.model_predictions = []


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        self.model_predictions = self.packer.pack(self.model_predictions,
                                                  inputs["image_id"],
                                                  inputs,
                                                  inputs["image_size"])

        if inputs["pipeline_end"] is True:

            # coco_instance = inputs["coco_instance"]
                
            cocoDt = self.coco_instance.loadRes(self.model_predictions)

            eval_type = 'bbox' if self.evaluation_type == 'detection' else self.evaluation_type
            cocoEval = COCOeval(self.coco_instance, cocoDt, eval_type)
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        return {}
