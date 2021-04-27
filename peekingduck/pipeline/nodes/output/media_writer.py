"""Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import os
from typing import Any, Dict
import numpy as np
import cv2
from peekingduck.pipeline.nodes.node import AbstractNode

# role of this node is to be able to take in multiple frames, stitch them together and output them.
# to do: need to have 'live' kind of data when there is no filename
# to do: it will be good to have the accepted file format as a configuration
# to do: somewhere so that input and output can use this config for media related issues


class Node(AbstractNode):
    """Node that processes videos and images as primary source input"""
    def __init__(self, config):
        super().__init__(config, node_path=__name__)

        self._file_name = None
        self._output_dir = config["outputdir"]
        self._prepare_directory(config["outputdir"])
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        #initialize for use in run
        self._file_name = None
        self._file_path = None
        self._image_type = None
        self.writer = None

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Writes media information to filepath

        Args:
            inputs: ["filename", "img", "fps"]

        Returns:
            outputs: [None]
        """

        if not self._file_name:
            self._prepare_writer(inputs["filename"],
                                 inputs["img"],
                                 inputs["fps"])

        if inputs["filename"] != self._file_name:
            self._prepare_writer(inputs["filename"],
                                 inputs["img"],
                                 inputs["fps"])
        else:
            self._write(inputs["img"])

        return {}

    def _write(self, img: np.array):
        if self._image_type == "image":
            cv2.imwrite(self._file_path, img)
        else:
            self.writer.write(img)

    def _prepare_writer(self, filename: str, img: np.array, fps: int):

        self._file_name = filename
        self._image_type = "video"
        if filename.split(".")[-1] in ["jpg", "jpeg", "png"]:
            self._image_type = "image"

        resolution = img.shape[1], img.shape[0]
        self._file_path = os.path.join(self._output_dir, filename)
        self.writer = cv2.VideoWriter(self._file_path, self._fourcc, fps, resolution)

    @staticmethod
    def _prepare_directory(outputdir):
        os.makedirs(outputdir, exist_ok=True)
