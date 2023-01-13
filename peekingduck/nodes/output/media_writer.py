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
Writes the output image/video to file.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from peekingduck.nodes.abstract_node import AbstractNode

# role of this node is to be able to take in multiple frames, stitch them
# together and output them.
# to do: need to have 'live' kind of data when there is no filename
# to do: it will be good to have the accepted file format as a configuration
# to do: somewhere so that input and output can use this config for media related issues


class Node(AbstractNode):
    """Outputs the processed image or video to a file. A timestamp is appended to the
    end of the file name.

    Inputs:
        |img_data|

        |filename_data|

        |saved_video_fps_data|

        |pipeline_end_data|

    Outputs:
        |none_output_data|

    Configs:
        output_dir (:obj:`str`): **default = "PeekingDuck/data/output"**. |br|
            Output directory for files to be written locally.
        output_filename (:obj:`str`): **default = null**. |br|
            Output filename for files to be written locally. Video only.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.output_dir: Path = Path(self.output_dir)  # type: ignore
        self.output_filename: Optional[str] = getattr(self, "output_filename", None)
        self._file_name: Optional[str] = None
        self._file_path_post_processed: Optional[str] = None
        self._input_type: Optional[str] = None
        self._frame: int = 0
        self.writer: None = None
        self._prepare_directory(self.output_dir)
        self._output_type: Optional[
            str
        ] = None  # video/img output from _output_filename
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.logger.info(f"Output directory used is: {self.output_dir}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes media information to filepath."""
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            if self.writer:  # images automatically releases writer
                self.writer.release()
            return {}
        # init
        self._input_type = self._detect_media_type(inputs["filename"])
        if self.output_filename is not None:
            output_filename: Path = Path(self.output_filename)
            # Use input_filename extension if output_filename extension is empty
            if output_filename.suffix == "":
                self.output_filename = str(
                    output_filename.with_suffix(Path(inputs["filename"]).suffix)
                )

            self._output_type = self._detect_media_type(self.output_filename)

        # different input file
        if self._file_name != inputs["filename"]:
            self._file_name = inputs["filename"]

            self._prepare_writer(
                inputs["filename"],
                inputs["img"],
                inputs["saved_video_fps"],
                self.output_filename,
            )
        self._write(inputs["img"])

        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"output_dir": str}

    def _write(self, img: np.ndarray) -> None:
        if self._input_type == "image" or self._output_type == "image":
            if self._input_type == "video":
                _file_path_post_processed: str = self._append_frame_filename(
                    self.output_filename  # only video with output_filename specified
                )
                cv2.imwrite(_file_path_post_processed, img)
            else:  # image input
                cv2.imwrite(self._file_path_post_processed, img)
        else:
            self.writer.write(img)

    def _prepare_writer(
        self,
        filename: str,
        img: np.ndarray,
        saved_video_fps: int,
        output_filename: Optional[str],
    ) -> None:
        # Use datetime stamp for image input
        self._file_path_post_processed = self._append_datetime_filename(filename)
        if self._input_type == "video":
            if output_filename is not None:
                self._file_path_post_processed = self._append_frame_filename(
                    output_filename
                )

            resolution = img.shape[1], img.shape[0]
            self.writer = cv2.VideoWriter(
                self._file_path_post_processed,
                self._fourcc,
                saved_video_fps,
                resolution,
            )

    @staticmethod
    def _prepare_directory(output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_media_type(filename: str) -> str:
        if Path(filename).suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return "image"

        return "video"

    def _append_datetime_filename(self, filename: str) -> str:
        current_time = datetime.datetime.now()
        # output as 'YYYYMMDD_hhmmss'
        time_str: str = current_time.strftime("%y%m%d_%H%M%S")

        # append timestamp to filename before extension Format: filename_timestamp.extension
        p_filename: Path = Path(filename)
        filename_with_timestamp: str = (
            f"{p_filename.stem}_{time_str}{p_filename.suffix}"
        )
        file_path_with_timestamp: Path = self.output_dir.joinpath(
            filename_with_timestamp
        )

        return str(file_path_with_timestamp)

    def _append_frame_filename(self, filename: Optional[str]) -> str:
        """append frame to filename before extension Format: filename_frame.extension"""
        assert filename is not None, "filename cannot be None"
        p_filename: Path = Path(filename)
        filename_with_frame: str = (
            f"{p_filename.stem}_{self._frame:05d}{p_filename.suffix}"
        )
        file_path_with_frame: Path = self.output_dir.joinpath(filename_with_frame)
        self._frame += 1

        return str(file_path_with_frame)
