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
Reads video/images from a directory.
"""

from pathlib import Path
from typing import Any, Dict

from peekingduck.pipeline.nodes.input.utils.preprocess import resize_image
from peekingduck.pipeline.nodes.input.utils.read import (
    VideoThread,
    VideoNoThread,
)
from peekingduck.pipeline.nodes.node import AbstractNode


# pylint: disable=R0902
class Node(AbstractNode):
    """Receives videos/image as inputs.

    Inputs:
        |none|

    Outputs:
        |img|

        |pipeline_end|

        |filename|

        |saved_video_fps|

    Configs:
        resize (:obj:`Dict`):
            **default = { do_resizing: False, width: 1280, height: 720 }**. |br|
            Dimension of extracted image frame.
        input_dir (:obj:`str`): **default = "PeekingDuck/data/input"**. |br|
            The directory to look for recorded video files and images.
        mirror_image (:obj:`bool`): **default = False**. |br|
            Flag to set extracted image frame as mirror image of input stream.
        threading (:obj:`bool`): **default = False**. |br|
            Boolean to enable threading when reading frames from input.
            The FPS may increase if this is enabled (system dependent).
        buffer_frames (:obj:`bool`): **default = False**. |br|
            Boolean to indicate if threaded class should buffer image frames.
            If threading is True, it is highly recommended that buffer_frames is
            also True to avoid losing frames, as otherwise the input thread would
            very likely read ahead of the main thread. |br|
            For more info, please refer to `input.recorded configuration
            <https://github.com/aimakerspace/PeekingDuck/blob/dev/peekingduck/configs/input/recorded.yml>`_.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._allowed_extensions = [
            "jpg",
            "jpeg",
            "png",
            "mp4",
            "avi",
            "mov",
            "mkv",
        ]
        self.file_end = False
        self.frame_counter = -1
        self.tens_counter = 10
        self._get_files(Path(self.input_dir))
        self._get_next_input()

        width, height = self.videocap.resolution
        self.logger.info(f"Video/Image size: {width} by {height}")
        if self.resize["do_resizing"]:
            self.logger.info(
                f"Resizing of input set to {self.resize['width']} "
                f"by {self.resize['height']}"
            )

        self.logger.info(f"Filepath used: {self.input_dir}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        input: ["none"],
        output: ["img", "pipeline_end"]
        """
        outputs = self._run_single_file()

        approx_processed = round((self.frame_counter / self.videocap.frame_count) * 100)
        self.frame_counter += 1

        if approx_processed > self.tens_counter:
            self.logger.info(f"Approximately Processed: {self.tens_counter}% ...")
            self.tens_counter += 10

        if self.file_end:
            self.logger.info(f"Completed processing file: {self._file_name}")
            pct_complete = round(100 * self.frame_counter / self.videocap.frame_count)
            self.logger.debug(f"#frames={self.frame_counter}, done={pct_complete}%")
            self._get_next_input()
            outputs = self._run_single_file()
            self.frame_counter = 0
            self.tens_counter = 10

        return outputs

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        self.file_end = True
        outputs = {
            "img": None,
            "pipeline_end": True,
            "filename": self._file_name,
            "saved_video_fps": self._fps,
        }
        if success:
            self.file_end = False
            if self.resize["do_resizing"]:
                img = resize_image(img, self.resize["width"], self.resize["height"])
            outputs = {
                "img": img,
                "pipeline_end": False,
                "filename": self._file_name,
                "saved_video_fps": self._fps,
            }

        return outputs

    def _get_files(self, path: Path) -> None:
        self._filepaths = [path]

        if path.is_dir():
            self._filepaths = list(path.iterdir())
            self._filepaths.sort()

        if not path.exists():
            raise FileNotFoundError("Filepath does not exist")
        if not self._filepaths:
            raise FileNotFoundError("No Media files available")

    def _get_next_input(self) -> None:
        if self._filepaths:
            file_path = self._filepaths.pop(0)
            self._file_name = file_path.name

            if self._is_valid_file_type(file_path):
                if getattr(self, "threading", False):
                    self.videocap = VideoThread(  # type: ignore
                        str(file_path), self.mirror_image, self.buffer_frames
                    )
                else:
                    self.videocap = VideoNoThread(  # type: ignore
                        str(file_path), self.mirror_image
                    )
                self._fps = self.videocap.fps
            else:
                self.logger.warning(
                    f"Skipping '{file_path}' as it is not an accepted "
                    f"file format {str(self._allowed_extensions)}"
                )
                self._get_next_input()

    def _is_valid_file_type(self, filepath: Path) -> bool:
        if filepath.suffix[1:] in self._allowed_extensions:
            return True
        return False

    def release_resources(self) -> None:
        """Override base class method to free video resource"""
        self.videocap.shutdown()
