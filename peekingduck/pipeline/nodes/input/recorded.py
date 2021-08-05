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
Reads video/images from a directory
"""

import os
from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.preprocess import resize_image
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread


# pylint: disable=R0902
class Node(AbstractNode):
    """Node to receive videos/image as inputs.

        Inputs:
            None

        Outputs:
            |img|

            |pipeline_end|

            |filename|

            |saved_video_fps|

        Configs:
            resize (:obj:`Dict`): **default = { do_resizing: False, width: 1280, height: 720 }**

                Dimension of extracted image frame

            input_dir (:obj: `str`): **default = 'PeekingDuck/data/input'**

                The directory to look for recorded video files and images

            mirror_image (:obj:`bool`): **default = False**

                Boolean to set extracted image frame as mirror image of input stream

    """

    def __init__(self, config: Dict[str, Any]=None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._allowed_extensions = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        self.file_end = False
        self.frame_counter = -1
        self.tens_counter = 10
        self._get_files(self.input_dir)
        self._get_next_input()

        width, height = self.videocap.resolution
        self.logger.info('Video/Image size: %s by %s',
                         width,
                         height)
        if self.resize['do_resizing']:
            self.logger.info('Resizing of input set to %s by %s',
                             self.resize['width'],
                             self.resize['height'])

        self.logger.info('Filepath used: %s', self.input_dir)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        '''
        input: ["none"],
        output: ["img", "pipeline_end"]
        '''
        outputs = self._run_single_file()

        approx_processed = round((self.frame_counter/self.videocap.frame_count)*100)
        self.frame_counter += 1

        if approx_processed > self.tens_counter:
            self.logger.info('Approximately Processed: %s%%...', self.tens_counter)
            self.tens_counter += 10

        if self.file_end:
            self.logger.info('Completed processing file: %s', self._file_name)
            self._get_next_input()
            outputs = self._run_single_file()
            self.frame_counter = 0
            self.tens_counter = 10

        return outputs

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        self.file_end = True
        outputs = {"img": None,
                   "pipeline_end": True,
                   "filename": self._file_name,
                   "saved_video_fps": self._fps}
        if success:
            self.file_end = False
            if self.resize['do_resizing']:
                img = resize_image(img,
                                   self.resize['width'],
                                   self.resize['height'])
            outputs = {"img": img,
                       "pipeline_end": False,
                       "filename": self._file_name,
                       "saved_video_fps": self._fps}

        return outputs

    def _get_files(self, path: str) -> None:
        self._filepaths = [path]

        if os.path.isdir(path):
            self._filepaths = os.listdir(path)
            self._filepaths = [os.path.join(path, filepath)
                               for filepath in self._filepaths]
            self._filepaths.sort()

        if not os.path.exists(path):
            raise FileNotFoundError("Filepath does not exist")
        if not self._filepaths:
            raise FileNotFoundError("No Media files available")

    def _get_next_input(self) -> None:

        if self._filepaths:
            file_path = self._filepaths.pop(0)
            self._file_name = os.path.basename(file_path)

            if self._is_valid_file_type(file_path):
                self.videocap = VideoNoThread(
                    file_path,
                    self.mirror_image
                )
                self._fps = self.videocap.fps
            else:
                self.logger.warning("Skipping '%s' as it is not an accepted file format %s",
                                    file_path,
                                    str(self._allowed_extensions)
                                    )
                self._get_next_input()

    def _is_valid_file_type(self, filepath: str) -> bool:

        if filepath.split(".")[-1] in self._allowed_extensions:
            return True
        return False
