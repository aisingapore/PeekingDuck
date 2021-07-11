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
Reader functions for input nodes
"""

from typing import Any, Tuple, Union
from threading import Thread, Lock
import cv2
from peekingduck.pipeline.nodes.input.utils.preprocess import mirror


class VideoThread:
    '''
    Videos will be threaded to prevent I/O blocking from affecting FPS.
    '''

    def __init__(self, input_source: str, mirror_image: bool) -> None:
        self.stream = cv2.VideoCapture(input_source)
        self.mirror = mirror_image
        if not self.stream.isOpened():
            raise ValueError("Camera or video input not detected: %s" % input_source)

        self._lock = Lock()

        self.frame = None
        thread = Thread(target=self._reading_thread, args=(), daemon=True)
        thread.start()

    def __del__(self) -> None:
        self.stream.release()

    def _reading_thread(self) -> None:
        '''
        A thread that continuously polls the camera for frames.
        '''
        while True:
            _, self.frame = self.stream.read()

    def read_frame(self) -> Union[bool, Any]:
        '''
        Reads the frame.
        '''
        self._lock.acquire()
        if self.frame is not None:
            frame = self.frame.copy()
            self._lock.release()
            if self.mirror:
                frame = mirror(frame)
            return True, frame

        self._lock.release()
        return False, None

    @property
    def resolution(self) -> Tuple[int, int]:
        """ Get resolution of the camera device used.

        Returns:
            width(int): width of input resolution
            height(int): heigh of input resolution
        """
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)


class VideoNoThread:
    '''
    No threading to deal with recorded videos and images.
    '''

    def __init__(self, input_source: str, mirror_image: bool) -> None:
        self.stream = cv2.VideoCapture(input_source)
        self.mirror = mirror_image
        if not self.stream.isOpened():
            raise ValueError("Video or image path incorrect: %s" % input_source)

    def __del__(self) -> None:
        self.stream.release()

    def read_frame(self) -> None:
        '''
        Reads the frame.
        '''
        return self.stream.read()

    @property
    def fps(self) -> float:
        """ Get FPS of videofile

        Returns:
            int: number indicating FPS
        """
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        return fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """ Get resolution of the file.

        Returns:
            width(int): width of resolution
            height(int): heigh of resolution
        """
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    @property
    def frame_count(self) -> int:
        """ Get total number of frames of file

        Returns:
            int: number indicating frame count
        """
        num_frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(num_frames)
