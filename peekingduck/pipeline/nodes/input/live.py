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
Reads a videofeed from a stream (e.g. webcam)
"""

from typing import Dict, Any

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.preprocess import resize_image
from peekingduck.pipeline.nodes.input.utils.read import VideoThread


class Node(AbstractNode):
    """Node to receive livestream as inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)

        self.resize_info = config['resize']
        input_source = config['input_source']
        mirror_image = config['mirror_image']
        self.fps_saved_output_video = config["fps_saved_output_video"]
        self.filename = config["filename"]

        self.videocap = VideoThread(input_source, mirror_image)

        width, height = self.videocap.resolution
        self.logger.info('Device resolution used: %s by %s', width, height)
        if self.resize_info['do_resizing']:
            self.logger.info('Resizing of input set to %s by %s',
                             self.resize_info['width'],
                             self.resize_info['height'])
        self.frame_counter = 0
        self.frames_log_freq = config['frames_log_freq']

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        if success:
            if self.resize_info['do_resizing']:
                img = resize_image(img,
                                   self.resize_info['width'],
                                   self.resize_info['height'])
            outputs = { "img": img,
                        "pipeline_end": False,
                        "filename": self.filename,
                        "fps": self.fps_saved_output_video}
            self.frame_counter += 1
            if self.frame_counter%self.frames_log_freq == 0:
                self.logger.info('Frames Processed: %s ...', self.frame_counter)

        else:
            outputs = { "img": None,
                        "pipeline_end": True,
                        "filename": self.filename,
                        "fps": self.fps_saved_output_video}
            self.logger.warning("No video frames available for processing.")

        return outputs
