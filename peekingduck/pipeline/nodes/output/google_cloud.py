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

"""
Saves an image to a Google Cloud Storage bucket.
"""

from typing import Any, Dict

import cv2
from google.cloud import storage
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(self.bucket_name)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        filename = inputs["filename"]
        if self.folder_name:
            blob = self.bucket.blob(self.folder_name + "/" + filename)
        else:
            blob = self.bucket.blob(filename)
        img_str = cv2.imencode(".jpg", inputs["img"])[1].tostring()
        blob.upload_from_string(img_str)

        return {}
