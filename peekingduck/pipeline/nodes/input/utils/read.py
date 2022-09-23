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
Reader functions for input nodes
"""

import http.client
import logging
import platform
import queue
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event, Thread
from typing import Any, Tuple, Union

import cv2

from peekingduck.pipeline.nodes.input.utils.png_reader import PNGReader
from peekingduck.pipeline.nodes.input.utils.preprocess import mirror

GOOGLE_DNS = "8.8.8.8"


def has_internet() -> bool:
    """Checks for internet connectivity by making a HEAD request to one of
    Google's public DNS servers.
    """
    # Suppress bandit B309 as PeekingDuck is meant to run on Python >= 3.6
    connection = http.client.HTTPSConnection(GOOGLE_DNS, timeout=5)  # nosec
    try:
        connection.request("HEAD", "/")
        return True
    except Exception:  # pylint: disable=broad-except
        return False
    finally:
        connection.close()


class VideoReader(ABC):
    """Class to read in videos and images."""

    def __init__(self, input_source: Union[int, str], mirror_image: bool) -> None:
        assert isinstance(input_source, (int, str))
        if isinstance(input_source, int):
            if platform.system().startswith("Windows"):
                # to eliminate opencv's "[WARN] terminating async callback" on Windows
                self.stream = cv2.VideoCapture(input_source, cv2.CAP_DSHOW)
            else:
                self.stream = cv2.VideoCapture(input_source)
        elif Path(input_source.lower()).suffix == ".png":
            self.stream = PNGReader(input_source)
        else:
            self.stream = cv2.VideoCapture(input_source)
        self._frame_counter = 0
        self.logger = logging.getLogger(type(self).__name__)
        self.mirror = mirror_image
        if not self.stream.isOpened():
            if self.is_url(input_source) and not has_internet():
                self.logger.warning("Possible network connectivity error.")
            raise ValueError(f"Video or image path incorrect: {input_source}")

    def __del__(self) -> None:
        # Note: self.logger.debug below crashes on Nvidia Jetson Xavier Ubuntu 18.04 python 3.6
        #       but does not crash on Intel MacBook Pro Ubuntu 20.04 python 3.7
        # self.logger.debug("__del__")
        self.stream.release()

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Any]:
        """Reads the frame."""

    def shutdown(self) -> None:
        """Shuts down this class.
        Cannot be merged into __del__ as threading code needs to run here.
        Dummy method left here for consistency with VideoThread class.
        """
        self.logger.debug("shutdown")

    @property
    @abstractmethod
    def queue_size(self) -> int:
        """Get buffer queue size

        Returns:
            int: number of frames in buffer
        """

    @property
    def fps(self) -> float:
        """Get FPS of videofile

        Returns:
            int: number indicating FPS
        """
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        return fps

    @property
    def frame_count(self) -> int:
        """Get total number of frames of file

        Returns:
            int: number indicating frame count
        """
        num_frames = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(num_frames)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get resolution of the file.

        Returns:
            width(int): width of resolution
            height(int): heigh of resolution
        """
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    @staticmethod
    def is_url(source: Union[int, str]) -> bool:
        """Checks if the provided ``source`` is a URL."""
        return isinstance(source, str) and "://" in source


class VideoThread(VideoReader):
    """
    Videos will be threaded to improve FPS by reducing I/O blocking latency.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, input_source: Union[int, str], mirror_image: bool, buffering: bool
    ) -> None:
        super().__init__(input_source, mirror_image)
        self.logger = logging.getLogger(type(self).__name__)
        # events to coordinate threading
        self.is_done = Event()
        self.is_thread_start = Event()
        # frame storage and buffering
        self.frame = None
        self.prev_frame = None
        self.buffer = buffering
        self.queue: queue.Queue = queue.Queue()
        # start threading
        self.thread = Thread(target=self._reading_thread, args=(), daemon=True)
        self.thread.start()
        self.is_thread_start.wait()

    def shutdown(self) -> None:
        """
        Shuts down this class.
        Cannot be merged into __del__ as threading code needs to run here.
        """
        self.logger.debug("VideoThread.shutdown")
        self.is_done.set()
        self.thread.join()

    def _reading_thread(self) -> None:
        """
        A thread that continuously polls the camera for frames.
        """
        while not self.is_done.is_set():
            if self.stream.isOpened():
                ret, frame = self.stream.read()
                if not ret:
                    self.logger.debug(
                        f"_reading_thread: ret={ret}, "
                        f"#frames read={self._frame_counter}"
                    )
                    self.is_done.set()
                else:
                    if self.mirror:
                        frame = mirror(frame)
                    self.frame = frame
                    self.is_thread_start.set()  # thread really started
                    self._frame_counter += 1
                    if self.buffer:
                        self.queue.put(self.frame)

    def read_frame(self) -> Tuple[bool, Any]:
        """
        Reads the frame.
        """
        # pylint: disable=no-else-return
        if self.buffer:
            if self.queue.empty():
                if self.is_done.is_set():
                    # end of input
                    return False, None
                else:
                    # input slow, so duplicate frame
                    return True, self.prev_frame
            else:
                self.prev_frame = self.queue.get()
                return True, self.prev_frame
        else:
            if self.is_done.is_set():
                return False, None
            else:
                return True, self.frame

    @property
    def queue_size(self) -> int:
        """Get buffer queue size

        Returns:
            int: number of frames in buffer
        """
        return self.queue.qsize()


class VideoNoThread(VideoReader):
    """
    No threading to deal with recorded videos and images.
    """

    def read_frame(self) -> Tuple[bool, Any]:
        """
        Reads the frame.
        """
        ret, frame = self.stream.read()
        if not ret:
            self.logger.debug(
                f"read_frame: ret={ret}, #frames read={self._frame_counter}"
            )
        else:
            if self.mirror:
                frame = mirror(frame)
            self._frame_counter += 1
        return ret, frame

    @property
    def queue_size(self) -> int:
        """Get buffer queue size

        Returns:
            int: number of frames in buffer
        """
        return 0
