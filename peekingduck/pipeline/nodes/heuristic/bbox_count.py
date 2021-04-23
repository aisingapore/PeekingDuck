"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict) -> None:
        super().__init__(config, name='heuristic.bbox_to_pt')

    def run(self, inputs: Dict) -> int:
        """Counts bboxes of object chosen in the frame. Note that this method
        requires that the bbox returns all the same objects (for example, all people)

        Args:
            bboxes (List[List[float]]): List of bboxes, each a list of 4 elements that
            defines the bounding box 

        Returns:
            count (int): count of number of same object within the image

        """
        return {'count': len(inputs[self.inputs[0]])}