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
Performs multiple object tracking for detected bboxes
"""

from typing import Dict, Any
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.dabble.utils.load_tracker import TrackerLoader


class Node(AbstractNode):
    """Node that uses bounding boxes detected by an object detector model
    to track multiple objects.

    There are types of trackers that can be selected: MOSSE, IOU.
    Please view each tracker's script, or the "Multi Object
    Tracking" use case documentation for more details.

    Inputs:
        |bboxes|

        |bbox_scores|

        |bbox_labels|

    Outputs:
        |obj_tags|

    Configs:
        tracking_type (:obj:`str`): **{"iou", "mosse"}, default="iou"**. |br|
            Type of tracking algorithm to be used. For more information
            about the trackers, please view the use case documentation.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.tracker = TrackerLoader(self.tracking_type)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Runs object tracking.

        Args:
            inputs (Dict[str, Any]): Dict of outputs from earlier nodes.

        Returns:
            Dict[str, Any]: Tracking ids of bounding boxes and ordered
                accordingly.
        """
        obj_tags = self.tracker.predict(inputs)
        return {"obj_tags": obj_tags}
