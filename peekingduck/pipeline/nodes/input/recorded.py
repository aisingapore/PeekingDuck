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
    """Node to receive videos/image as inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, node_path=__name__)
        self._allowed_extensions = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        input_dir = config['input_dir']
        self.resize_info = config['resize']
        self._mirror_image = config['mirror_image']
        self.file_end = False
        self.frame_counter = -1
        self.tens_counter = 10

        self._get_files(input_dir)
        self._get_next_input()

        width, height = self.videocap.resolution
        self.logger.info('Video/Image size: %s by %s',
                         width,
                         height)
        if self.resize_info['do_resizing']:
            self.logger.info('Resizing of input set to %s by %s',
                             self.resize_info['width'],
                             self.resize_info['height'])

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
                   "fps": self._fps}
        if success:
            self.file_end = False
            if self.resize_info['do_resizing']:
                img = resize_image(img,
                                   self.resize_info['width'],
                                   self.resize_info['height'])
            outputs = {"img": img,
                       "pipeline_end": False,
                       "filename": self._file_name,
                       "fps": self._fps}

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
                    self._mirror_image
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
