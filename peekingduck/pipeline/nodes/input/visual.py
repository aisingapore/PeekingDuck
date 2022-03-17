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
Reads inputs from multiple visual sources
- image or video file on local storage
- folder of images or videos
- online cloud source
- CCTV or webcam live feed
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from peekingduck.pipeline.nodes.input.utils.preprocess import resize_image
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread, VideoThread
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-many-instance-attributes
    """Receives visual sources as inputs.

    Inputs:
        |none_input_data|

    Outputs:
        |img_data|

        |filename_data|

        |pipeline_end_data|

        |saved_video_fps_data|

    Configs:
        filename (:obj:`str`): **default = "video.mp4"**. |br|
            Filename of the MP4 file if media is exported. |br|
            If source is a directory of files, then filename is the current file being
            processed.
        frames_log_freq (:obj:`int`): **default = 100**. |br|
            Logs frequency of frames passed in CLI
        mirror_image (:obj:`bool`): **default = False**. |br|
            Flag to set extracted image frame as mirror image of input stream.
        resize (:obj:`Dict`):
            **default = { do_resizing: False, width: 1280, height: 720 }** |br|
            Dimension of extracted image frame.
        saved_video_fps (:obj:`int`): **default = 10**. |br|
            FPS to be used to output the MP4 file after livestream is processed and
            exported.  FPS is dependent on running machine performance.
        source (:obj:`int`):
            **default = https://storage.googleapis.com/peekingduck/videos/wave.mp4**. |br|
            Input source can be: |br|
            - filename : local image or video file |br|
            - directory name : all media files will be processed |br|
            - http URL for online cloud source : http[s]://... |br|
            - rtsp URL for CCTV : rtsp://... |br|
            - 0 for webcam live feed |br|
            Refer to `OpenCV documentation
            <https://docs.opencv.org/4.5.5/d8/dfe/classcv_1_1VideoCapture.html>`_
            for more technical information.
        threading (:obj:`bool`): **default = False**. |br|
            Flag to enable threading when reading frames from camera.
            The FPS can increase up to 30%.
        buffer_frames (:obj:`bool`): **default = False**. |br|
            Boolean to indicate if threaded class should buffer image frames.
            If threading is True and output.media_writer is enabled, then
            buffer_frames should be True to ensure output video is correctly
            saved. One side effect of threading=True, buffer_frames=True is the
            onscreen video display could appear laggy due to the buffering.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._allowed_extensions = [
            "gif",
            "jpeg",
            "jpg",
            "png",
            "avi",
            "m4v",
            "mkv",
            "mov",
            "mp4",
        ]
        self._file_name: str = ""
        self._fps: float = 0  # self._fps > 0 if file playback
        self.frame_counter: int = 0
        self.tens_counter: int = 10
        self.total_frame_count: int = 0
        self.progress: int = 0
        self.videocap: Optional[Union[VideoNoThread, VideoThread]] = None
        self.do_resize = self.resize["do_resizing"]
        self.has_multiple_inputs = self._source_is_directory()

        print("*****", self.filename)
        if not self._is_valid_file_type(Path(self.filename)):
            raise ValueError(
                f"filename extension must be one of: {self._allowed_extensions}"
            )

        if self.has_multiple_inputs:
            self._get_files(Path(self.source))
            self._get_next_file()
        else:
            self._open_input(self.source)

        if self.videocap:
            # check resizing configuration
            width, height = self.videocap.resolution
            self.logger.info(f"Input size: {width} by {height}")
            if self.do_resize:
                self.logger.info(
                    f"Resizing of input set to {self.resize['width']} "
                    f"by {self.resize['height']}"
                )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run this node

        Args:
            inputs (Dict[str, Any]): ["img"]

        Returns:
            Dict[str, Any]: ["img", "pipeline_end", "saved_video_fps"]
        """
        outputs = self._get_next_frame()
        self._show_progress()
        # chain next input if available
        if self.file_end and self.has_multiple_inputs:
            self.logger.info(f"Completed processing file: {self._file_name}")
            self.logger.debug(f"#frames={self.frame_counter}, done={self.progress}%")
            self._get_next_file()
            outputs = self._get_next_frame()
            self.frame_counter = 0
            self.tens_counter = 10
        return outputs

    def _get_next_frame(self) -> Dict[str, Any]:
        """Read next frame from current input file/source"""
        self.file_end = True
        outputs = {
            "img": None,
            "filename": self._file_name if self._file_name else self.filename,
            "pipeline_end": True,
            "saved_video_fps": self._fps if self._fps > 0 else self.saved_video_fps,
        }
        if self.videocap:
            success, img = self.videocap.read_frame()
            if success:
                self.file_end = False
                if self.do_resize:
                    img = resize_image(img, self.resize["width"], self.resize["height"])
                outputs["img"] = img
                outputs["pipeline_end"] = False
            else:
                self.logger.debug("No video frames available for processing.")
        return outputs

    def _get_files(self, path: Path) -> None:
        """Read all files in given directory

        Args:
            path (Path): the directory path

        Raises:
            FileNotFoundError: directory does not exist error
        """
        if not path.exists():
            raise FileNotFoundError("Filepath does not exist")

        self._filepaths = [path]
        self.logger.info(f"Directory: {path}")
        self._filepaths = list(path.iterdir())
        self._filepaths.sort()

    def _get_next_file(self) -> None:
        """Load next file in a directory of files"""
        if self._filepaths:
            file_path = self._filepaths.pop(0)
            self._file_name = file_path.name

            if self._is_valid_file_type(file_path):
                self._open_input(str(file_path))
            else:
                self.logger.warning(
                    f"Skipping '{file_path}' as it is not an accepted "
                    f"file format {str(self._allowed_extensions)}"
                )
                self._get_next_file()

    def _is_valid_file_type(self, filepath: Path) -> bool:
        """Check if given file has a supported file extension.

        Args:
            filepath (Path): the file to be file-type checked

        Returns:
            bool: True if supported file type else False
        """
        return filepath.suffix[1:] in self._allowed_extensions

    def _open_input(self, input_source: Any) -> None:
        """Open given input source for consumption.

        Args:
            input_source (Any): any of the following supported inputs
                                - image or video file on local storage
                                - folder of images or videos
                                - online cloud source
                                - CCTV or webcam live feed
        """
        if self.threading:
            self.videocap = VideoThread(
                self.source, self.mirror_image, self.buffer_frames
            )
        else:
            self.videocap = VideoNoThread(input_source, self.mirror_image)
        self._fps = self.videocap.fps
        self.total_frame_count = self.videocap.frame_count

    def _show_progress(self) -> None:
        """Show progress information during pipeline iteration"""
        self.frame_counter += 1
        if self.frame_counter % self.frames_log_freq == 0:
            self.logger.info(f"Frames Processed: {self.frame_counter} ...")
        if self.total_frame_count > 0:
            self.progress = round((self.frame_counter / self.total_frame_count) * 100)
            if self.progress >= self.tens_counter:
                self.logger.info(f"Progress: {self.tens_counter}% ...")
                self.tens_counter += 10

    def _source_is_directory(self) -> bool:
        """
        Check if input source is a directory of files or not.
        If yes, then node will have specific methods to handle it.
        If not, then opencv can deal with all non-directory sources.
        """
        is_url = str(self.source).startswith(("http://", "https://", "rtsp://"))
        if isinstance(self.source, int) or is_url:
            return False
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist")
        return path.is_dir()

    def release_resources(self) -> None:
        """Override base class method to free video resource"""
        if self.videocap:
            self.videocap.shutdown()
