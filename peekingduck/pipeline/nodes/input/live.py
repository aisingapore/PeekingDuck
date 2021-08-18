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
from peekingduck.pipeline.nodes.input.utils.read import VideoThread, VideoNoThread


class Node(AbstractNode):
    """Node to receive livestream as inputs.

    Inputs:
        None

    Outputs:
        |img|

        |filename|

        |pipeline_end|

        |saved_video_fps|

    Configs:
        fps_saved_output_video (:obj:`int`): **default = 10**

            FPS of the mp4 file after livestream is processed and exported.
            FPS dependent on running machine performance.

        filename (:obj:`str`):  **default = "webcamfeed.mp4"**

            Filename of the mp4 file if livestream is exported.

        resize (:obj:`Dict`): **default = { do_resizing: False, width: 1280, height: 720 }**

            Dimension of extracted image frame

        input_source (:obj:`int`): **default = 0 (for webcam)**

            Refer to `OpenCV doucmentation
            <https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d/>`_
            for list of source stream codes

        mirror_image (:obj:`bool`): **default = False**

            Boolean to set extracted image frame as mirror image of input stream

        frames_log_freq (:obj:`int`): **default = 100**

            Logs frequency of frames passed in cli

        threading (:obj:`bool`): **default = False**

            Boolean to enable threading when reading frames from camera.
            The FPS can increase up to 30% if this is enabled for Windows and MacOS.
            This will also be supported in Linux in future PeekingDuck versions.

    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._allowed_extensions = ["mp4", "avi", "mov", "mkv"]
        if self.threading:
            self.videocap = VideoThread(                # type: ignore
                self.input_source, self.mirror_image)
        else:
            self.videocap = VideoNoThread(              # type: ignore
                self.input_source, self.mirror_image)

        width, height = self.videocap.resolution
        self.logger.info('Device resolution used: %s by %s', width, height)
        if self.resize['do_resizing']:
            self.logger.info('Resizing of input set to %s by %s',
                             self.resize['width'],
                             self.resize['height'])
        if self.filename.split('.')[-1] not in self._allowed_extensions:
            raise ValueError("filename extension must be one of: ",
                             self._allowed_extensions)

        self.frame_counter = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        if success:
            if self.resize['do_resizing']:
                img = resize_image(img,
                                   self.resize['width'],
                                   self.resize['height'])

            outputs = {"img": img,
                       "pipeline_end": False,
                       "filename": self.filename,
                       "saved_video_fps": self.fps_saved_output_video}
            self.frame_counter += 1
            if self.frame_counter % self.frames_log_freq == 0:
                self.logger.info('Frames Processed: %s ...',
                                 self.frame_counter)

        else:
            outputs = {"img": None,
                       "pipeline_end": True,
                       "filename": self.filename,
                       "saved_video_fps": self.fps_saved_output_video}
            self.logger.warning("No video frames available for processing.")

        return outputs
