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
    """Saves an image with a given filename to Google Cloud Storage.

    Inputs:
        |img_data|

        |filename_data|

    Outputs:
        |none_output_data|

    Configs:
        bucket_name (:obj:`str`): **default = "peekingduck"** |br|
            Name of Google Cloud Storage bucket.
        folder_name (:obj:`str`): **default = null** |br|
            Name of folder within Google Cloud Storage bucket, if any. Subfolders should be
            separated by ``/``. ::

                # Examples of folder_name
                level_1
                level_1/level_2/level_3

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(self.bucket_name)
        if not self.bucket.exists():
            raise ValueError(
                f"The bucket: {self.bucket} does not exist on Google Cloud Storage."
            )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Saves an image with a given filename to Google Cloud Storage."""

        filename = inputs["filename"]
        if self.folder_name:
            blob = self.bucket.blob(self.folder_name + "/" + filename)
        else:
            blob = self.bucket.blob(filename)
        img_str = cv2.imencode(".jpg", inputs["img"])[1].tostring()
        blob.upload_from_string(img_str)

        return {}
