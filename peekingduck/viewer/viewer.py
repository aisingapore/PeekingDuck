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
Implement PeekingDuck Viewer
"""

from typing import List
from contextlib import redirect_stderr
from pathlib import Path
import logging
from io import StringIO
import os
import platform
import traceback
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import askyesno, showerror
import threading
import copy
import cv2
import numpy as np
from PIL import Image, ImageTk
from peekingduck.declarative_loader import DeclarativeLoader
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.viewer.playlist import PlayList
from peekingduck.viewer.viewer_gui import create_window
from peekingduck.viewer.viewer_utils import (
    get_keyboard_char,
    get_keyboard_modifier,
)

####################
# Globals
####################
BUTTON_DELAY: int = 250  # milliseconds (0.25 of a second)
BUTTON_REPEAT: int = int(1000 / 60)  # milliseconds (60 fps)
STOP_PIPELINE_DELAY: float = 2.0  # seconds before switching pipelines
FPS_60: int = int(1000 / 60)  # milliseconds per iteration
ZOOM_TEXT: List[str] = ["50%", "75%", "100%", "125%", "150%", "200%", "250%", "300%"]
ZOOM_DEFAULT_IDX: int = 2
ZOOMS: List[float] = [0.5, 0.75, 1.0, 1.25, 1.50, 2.00, 2.50, 3.00]  # > 3x is slow!
PLAY_BUTTON_TEXT = "Play"
STOP_BUTTON_TEXT = "Stop"


def parse_streams(io_stream: StringIO) -> str:
    """Helper method to parse I/O streams.
    Used to capture errors/exceptions from PeekingDuck.

    Args:
        io_stream (StringIO): the I/O stream to parse

    Returns:
        str: parsed stream
    """
    msg = io_stream.getvalue()
    msg = os.linesep.join([s for s in msg.splitlines() if s])
    return msg


class Viewer:  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """Implement PeekingDuck Viewer class"""

    def __init__(
        self,
        pipeline_path: Path,
        config_updates_cli: str,
        custom_nodes_parent_subdir: str,
        num_iter: int = 0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.config_updates_cli = config_updates_cli
        self.custom_nodes_parent_path = custom_nodes_parent_subdir
        self.num_iter = num_iter
        # init PlayList object
        self.home_path = Path.home()
        self.playlist = PlayList(self.home_path)
        self.playlist.load_playlist_file()
        self.playlist_show: bool = False
        # init pipeline to run
        self.init_pipeline(pipeline_path)
        self.playlist.add_pipeline(self.pipeline_full_path)
        # configure keyboard shortcuts map
        self._keyboard_shortcuts = {
            "=": self._zoom_reset,
            "+": self._zoom_in,
            "-": self._zoom_out,
        }
        # forward type hinting of internal working vars
        self._frames: List[np.ndarray] = []
        self._frame_idx: int = -1
        self.zoom_idx: int = ZOOM_DEFAULT_IDX
        self.is_output_playback: bool = False
        self.is_pipeline_running: bool = False
        self.state: str = "play"

    def run(self) -> None:
        """Main method to setup Viewer and run Tk event loop"""
        self.logger.debug(f"cwd={Path.cwd()}")
        self.logger.debug(f"pipeline={self.pipeline_path}")
        # create Tkinter window and frames
        create_window(self)
        # trap macOS Cmd-Q keystroke
        if platform.system() == "Darwin":
            self.logger.debug("binding macOS cmd-Q")
            self.root.createcommand("::tk::mac::Quit", self.on_exit)
        # activate internal timer function and start Tkinter event loop
        self.timer_function()
        self.root.mainloop()

    ####################
    #
    # Tk Event Handlers
    #
    ####################
    def btn_hide_show_playlist_press(self) -> None:
        """Handle Hide/Show Playlist button

        Tk technotes:
            - Behavior:
                Playlist on right is fixed width.  When image is expanded, it should not
                cover playlist.  But when playlist is hidden and revealed, the expanding
                image will cover playlist
            - To fix above, need to
              a) also pack_forget() the image frame, along with the playlist frame
              b) when revealing playlist, pack() playlist first, then pack() image
              c) now the expanding image will not cover the playlist
        """
        if self.playlist_show:
            self.tk_playlist_frm.pack_forget()
        else:
            self.tk_playlist_frm.pack(side=tk.RIGHT, fill=tk.Y)
            self.tk_playlist_view.reset()
            self.tk_playlist_view.select(str(self.pipeline_full_path))
        self.playlist_show = not self.playlist_show

    def btn_play_stop_press(self) -> None:
        """Handle Play/Stop button"""
        self.logger.debug(f"btn_play_stop_press start: self.state={self.state}")
        if self.is_pipeline_running:
            self.stop_running_pipeline()
        elif self.is_output_playback:
            self.stop_playback()
        else:
            self.start_playback()
        self.logger.debug(f"btn_play_stop_press end: self.state={self.state}")

    def btn_zoom_in_press(self) -> None:
        """Zoom in: make image larger"""
        self._zoom_in()

    def btn_zoom_out_press(self) -> None:
        """Zoom out: make image smaller"""
        self._zoom_out()

    def on_exit(self) -> None:
        """Handle viewer quit event"""
        self.logger.info("quitting viewer")

        if self.is_pipeline_running or self.is_output_playback:
            self.logger.debug("stopping current pipeline")
            if self.is_pipeline_running:
                self.run_pipeline_end()
            elif self.is_output_playback:
                self.stop_playback()

            # add non-blocking wait to let background task clean up properly
            self.logger.debug(f"wait {STOP_PIPELINE_DELAY} sec")
            wait_event = threading.Event()
            wait_event.wait(STOP_PIPELINE_DELAY)

        self.cancel_timer_function()
        self.logger.debug("saving playlist")
        self.playlist.save_playlist_file()
        self.root.destroy()

    def on_keypress(self, event: tk.Event) -> None:
        """Handle all keydown events.
        Default system shortcuts are automatically handled, e.g. CMD-Q quits on macOS

        Args:
            event (tk.Event): the key down event
        """
        self.logger.debug(
            f"keypressed: char={event.char}, keysym={event.keysym}, state={event.state}"
        )
        key_state: int = int(event.state)
        mod = get_keyboard_modifier(key_state)
        key = get_keyboard_char(event.char, event.keysym)
        self.logger.debug(f"mod={mod}, key={key}")
        # handle supported keyboard shortcuts here
        if mod.startswith("ctrl"):
            if key in self._keyboard_shortcuts:
                self._keyboard_shortcuts[key]()

    def on_resize(self, event: tk.Event) -> None:
        """Handle window resize event.

        Args:
            event (tk.Event): The resize event.
        """
        if str(event.widget) == ".":
            # NB: "." is the root widget, i.e. main window
            self.logger.debug(
                f"on_resize: widget={event.widget}, h={event.height}, w={event.width}"
            )

    def on_add_pipeline(self) -> bool:
        """Add pipeline to playlist

        Returns:
            bool: True if pipeline added, False otherwise
        """
        # filetypes = (("Pipeline files", "*.yml"), ("All files", "*.*"))
        filetypes = [("Pipeline files", "*.yml")]
        pipeline_filepath = filedialog.askopenfilename(
            title="Open a pipeline file (*.yml)",
            initialdir=self.home_path,
            filetypes=filetypes,
        )
        self.logger.debug(f"on add pipeline: filepath={pipeline_filepath}")
        if pipeline_filepath:
            self.playlist.add_pipeline(pipeline_filepath)
            return True
        return False

    def on_delete_pipeline(self, pipeline: str) -> bool:
        """Delete pipeline from playlist

        Args:
            pipeline (str): Pipeline to delete
        """
        self.logger.debug(f"on delete pipeline {pipeline}")
        answer = askyesno(
            title="Confirm Delete Pipeline from Playlist",
            message=f"Are you sure you want to delete\n\n{pipeline}\n\nfrom playlist?",
        )
        if answer:
            self.playlist.delete_pipeline(pipeline)
            return True
        return False

    def on_run_pipeline(self, pipeline: str) -> None:
        """Callback function for Run pipeline

        Args:
            pipeline (str): Pipeline to execute
        """
        # run new pipeline
        self.logger.debug(f"on run pipeline {pipeline}")
        if self.is_pipeline_running:
            self.run_pipeline_end()
        elif self.is_output_playback:
            self.stop_playback()

        # add non-blocking wait to let background task clean up properly
        self.logger.debug(f"wait {STOP_PIPELINE_DELAY} sec")
        wait_event = threading.Event()
        wait_event.wait(STOP_PIPELINE_DELAY)

        # double check status variables
        self.logger.debug(f"self.state={self.state}")
        self.logger.debug(f"self.is_output_playback={self.is_output_playback}")
        self.logger.debug(f"self.is_pipeline_running={self.is_pipeline_running}")
        if self.state == "play" or self.is_output_playback or self.is_pipeline_running:
            self.logger.warning(
                "Inconsistent internal self states:\n"
                f"  state={self.state} - expect !play"
                f"  is_output_playback={self.is_output_playback} - expect False"
                f"  is_pipeline_running={self.is_pipeline_running} - expect False"
            )

        # start new pipeline here
        self.cancel_timer_function()
        self.init_pipeline(Path(pipeline))
        self.timer_function()

    ##################
    #
    # Background Timer
    #
    ##################
    def timer_function(self) -> None:
        """Function to do background processing in Tkinter's way"""
        if self.state == "play":
            # Only two states: 1) playing back video or 2) executing pipeline
            if self.is_output_playback:
                self.do_playback()
            else:
                # Executing pipeline: check which execution state we are in
                if not self.is_pipeline_running:
                    self.run_pipeline_start()
                elif self._pipeline.terminate:
                    self.run_pipeline_end()
                else:
                    self.run_pipeline_one_iteration()

        self.root.update()  # wake up GUI
        self._bkgd_job = self.tk_header.after(FPS_60, self.timer_function)

    def cancel_timer_function(self) -> None:
        """Cancel the background timer function"""
        if self._bkgd_job:
            self.tk_header.after_cancel(self._bkgd_job)
            self._bkgd_job = None

    ####################
    #
    # Display Management
    #
    ####################
    def _backward_one_frame(self) -> bool:
        """Move back one frame, can be called repeatedly"""
        if self._frame_idx > 0:
            self._frame_idx -= 1
            self._update_slider_and_show_frame()
            return True
        return False

    def _forward_one_frame(self) -> bool:
        """Move forward one frame, can be called repeatedly"""
        if self._frame_idx + 1 < len(self._frames):
            self._frame_idx += 1
            self._update_slider_and_show_frame()
            return True
        return False

    def _show_frame(self) -> None:
        """Display image frame pointed to by frame_idx"""
        if self._frames:
            frame = self._frames[self._frame_idx]
            frame = self._apply_zoom(frame)
            # self.logger.debug(f"show_frame {self.frame_idx} size={frame.shape}")
            img_arr = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img_arr)
            # self.logger.debug(f"img_tk: {img_tk.width()}x{img_tk.height()}")
            self._img_tk = img_tk  # save to avoid python GC
            self.tk_output_image.config(image=img_tk)

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Zoom output image according to current zoom setting

        Args:
            frame (np.ndarray): image frame data to be zoomed

        Returns:
            np.ndarray: the zoomed image
        """
        if self.zoom_idx != ZOOM_DEFAULT_IDX:
            zoom = ZOOMS[self.zoom_idx]
            new_size = (
                int(frame.shape[0] * zoom),
                int(frame.shape[1] * zoom),
                frame.shape[2],
            )
            frame = cv2.resize(frame, (new_size[1], new_size[0]))
        return frame

    def _zoom_in(self) -> None:
        """Zoom in on image"""
        if self.zoom_idx + 1 < len(ZOOMS):
            self.zoom_idx += 1
            self._update_zoom_and_show_frame()

    def _zoom_out(self) -> None:
        """Zoom out on image"""
        if self.zoom_idx > 0:
            self.zoom_idx -= 1
            self._update_zoom_and_show_frame()

    def _zoom_reset(self) -> None:
        """Reset zoom to default 1x"""
        self.zoom_idx = ZOOM_DEFAULT_IDX
        self._update_zoom_and_show_frame()

    def _update_zoom_and_show_frame(self) -> None:
        """Update zoom widget and refresh current frame"""
        glyph = ZOOM_TEXT[self.zoom_idx]
        self.tk_lbl_zoom["text"] = f"{glyph}"
        self._show_frame()

    def _set_header_playing(self) -> None:
        """Change header text to playing..."""
        self.tk_header["text"] = f"Playing {self.pipeline_path.name}"

    def _set_header_running(self) -> None:
        """Change header text to running..."""
        self.tk_header["text"] = f"Running {self.pipeline_path.name}"

    def _set_header_stop(self) -> None:
        """Change header text to pipeline pathname"""
        self.tk_header["text"] = f"{self.pipeline_path.name}"

    def _update_slider_and_show_frame(self) -> None:
        """Update slider based on frame index and show new frame"""
        frame_num = self._frame_idx + 1
        self.tk_scale.set(frame_num)
        self.tk_lbl_frame_num["text"] = frame_num
        self._show_frame()

    def _enable_progress(self) -> None:
        """Show progress bar and hide slider"""
        self.logger.debug("enable progress")
        self.tk_scale.grid_remove()  # hide slider
        self.tk_progress.grid()  # show progress bar

    def _enable_slider(self) -> None:
        """Show slider and hide progress bar"""
        self.logger.debug("enable slider")
        self.tk_progress.grid_remove()  # hide progress bar
        self.tk_scale.grid()  # show slider
        self.tk_scale.configure(from_=1, to=len(self._frames))
        self.sync_slider_to_frame(self.tk_scale.get())

    def _set_status_text(self, text: str) -> None:
        """Set the status bar text to given text string.

        Args:
            text (str): the status bar text
        """
        self.tk_status_bar["text"] = text

    def sync_slider_to_frame(self, val: str) -> None:
        """Update frame index based on slider value change

        Args:
            val (str): slider value
        """
        self.logger.debug(f"sync slider to frame: {val} {type(val)}")
        idx = int(self.tk_scale.get())
        idx_start = self.tk_scale["from"]
        idx_end = self.tk_scale["to"]
        self.logger.debug(f"idx={idx}, from:{idx_start}, to:{idx_end})")
        if idx >= len(self._frames):
            idx = len(self._frames)
        self._frame_idx = idx - 1
        self.tk_lbl_frame_num["text"] = idx
        self._show_frame()

    ####################
    #
    # Pipeline Execution
    #
    ####################
    def init_pipeline(self, pipeline_path: Path) -> None:
        """Initialise pipeline to be run

        Args:
            pipeline_path (Path): The pipeline to run
        """
        self.pipeline_path = pipeline_path
        is_abs = pipeline_path.is_absolute()
        self.logger.debug(
            f"init pipeline: pipeline_path={pipeline_path}, is_abs={is_abs}"
        )
        # expand pipeline path if required
        if not is_abs:
            full_path = pipeline_path.resolve()
            self.logger.debug(f"full path: {full_path}")
            self.pipeline_full_path = full_path
        else:
            self.pipeline_full_path = pipeline_path
        # set custom nodes path accordingly
        self.custom_nodes_parent_path = str(self.pipeline_full_path.parent / "src")
        self.logger.debug(f"custom nodes parent: {type(self.custom_nodes_parent_path)}")
        # init internal working vars
        self._frames = []
        self._frame_idx = -1
        self.zoom_idx = ZOOM_DEFAULT_IDX
        self.is_output_playback = False
        self.is_pipeline_running = False
        self.state = "play"  # activate auto play (cf. self.timer_function)
        self.bkgd_job = None

    def pipeline_error(self, exc_msg: str, err_stream: StringIO) -> None:
        """Helper method to handle pipeline error conditions

        Args:
            exc_msg (str): message from exception object
            err_stream (StringIO): error I/O stream
        """
        self.logger.error("Error when running pipeline:")
        self.logger.error(f"Exception msg: {exc_msg}")
        err_msg = parse_streams(err_stream)
        self.logger.error(f"Error msg: {err_msg}")
        self.run_pipeline_end()
        self.set_viewer_state_to_stop()
        if "FileNotFoundError" in exc_msg:
            showerror(
                "Pipeline Runtime Error",
                f"Missing pipeline file:\n\n{self.pipeline_path}",
            )
        if "CUDA" in exc_msg:
            showerror("CUDA Runtime Error", "Please see logs for more details")

    def run_pipeline_end(self) -> None:
        """Called when pipeline execution is completed.
        To perform clean-up/housekeeping tasks to ensure system consistency"""
        self.logger.debug("run pipeline end")
        for node in self._pipeline.nodes:
            if node.name.endswith("input.visual"):
                node.release_resources()  # clean up nodes with threads
        self.is_pipeline_running = False
        self._enable_slider()
        self.set_viewer_state_to_stop()
        self._set_header_stop()

    def run_pipeline_one_iteration(self) -> None:  # pylint: disable=too-many-branches
        """Run one pipeline iteration"""
        self.is_pipeline_running = True
        err_runtime = False
        exc_msg = ""
        err_stream = StringIO()
        # technote: Detect runtime exception with flag as exception object holds ref to
        # error stack frame, preventing further objects from being freed.
        with redirect_stderr(err_stream):
            try:
                for node in self._pipeline.nodes:
                    if self._pipeline.data.get("pipeline_end", False):
                        self._pipeline.terminate = True
                        if "pipeline_end" not in node.inputs:
                            continue
                    if "all" in node.inputs:
                        inputs = copy.deepcopy(self._pipeline.data)
                    else:
                        inputs = {
                            key: self._pipeline.data[key]
                            for key in node.inputs
                            if key in self._pipeline.data
                        }
                    if hasattr(node, "optional_inputs"):
                        # Nodes won't receive inputs with optional key if not found upstream
                        for key in node.optional_inputs:
                            if key in self._pipeline.data:
                                inputs[key] = self._pipeline.data[key]
                    if node.name.endswith("output.screen"):
                        pass  # disable duplicate video from output.screen
                    else:
                        outputs = node.run(inputs)
                        self._pipeline.data.update(outputs)
                    # check for FPS on first iteration
                    if self._frame_idx == 0 and node.name.endswith("input.visual"):
                        num_frames = node.total_frame_count
                        if num_frames > 0:
                            self.tk_progress["maximum"] = num_frames
                        else:
                            self.tk_progress["mode"] = "indeterminate"
            except Exception:  # pylint: disable=broad-except
                err_runtime = True
                exc_msg = traceback.format_exc()

        # handle pipeline runtime error
        if err_runtime:
            self.pipeline_error(exc_msg, err_stream)
            return

        # render img into screen output to Tkinter
        img = self._pipeline.data["img"]
        if img is not None:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB for Tkinter
            self._frames.append(frame)  # save frame for playback
            self._frame_idx += 1
            self._show_frame()

        # update progress bar after each iteration
        self.tk_progress["value"] = self._frame_idx
        self.tk_lbl_frame_num["text"] = self._frame_idx + 1

        # check if need to stop after fixed number of iterations
        if self.num_iter and self._frame_idx + 1 >= self.num_iter:
            self.logger.info(f"Stopping pipeline after {self.num_iter} iterations")
            self.stop_running_pipeline()

    def run_pipeline_start(self) -> None:
        """Start PeekingDuck's pipeline"""
        self.logger.debug("run pipeline start")
        self.logger.debug(f"pipeline path: {self.pipeline_path}")
        self.logger.debug(f"custom_nodes: {self.custom_nodes_parent_path}")
        # technotes: node __init__() is called in get_pipeline() below.
        #            Detect runtime exception with flag and handle it outside as
        #            exception object holds reference to stack frame where error was
        #            raised, preventing memory from being freed.
        err_runtime = False
        exc_msg = ""
        err_stream = StringIO()
        with redirect_stderr(err_stream):
            try:
                self._node_loader = DeclarativeLoader(
                    self.pipeline_path,
                    self.config_updates_cli,
                    self.custom_nodes_parent_path,
                    pkd_viewer=True,
                )
                self._pipeline: Pipeline = self._node_loader.get_pipeline()
            except Exception:  # pylint: disable=broad-except
                err_runtime = True
                exc_msg = traceback.format_exc()

        # handle pipeline runtime error
        if err_runtime:
            self.pipeline_error(exc_msg, err_stream)
        else:
            self._set_header_running()
            self.set_viewer_state_to_play()
            self._set_status_text(f"Pipeline: {self.pipeline_full_path}")
            self.is_pipeline_running = True
            self._enable_progress()

    def stop_running_pipeline(self) -> None:
        """Signal pipeline execution to be stopped"""
        self._pipeline.terminate = True

    ###################
    #
    # Pipeline Playback
    #
    ###################
    def start_playback(self) -> None:
        """Start output playback process"""
        self.is_output_playback = True
        # auto-rewind if at last frame
        if self._frame_idx + 1 >= len(self._frames):
            self._frame_idx = 0
            self._update_slider_and_show_frame()
        self.set_viewer_state_to_play()
        self._set_header_playing()
        self.do_playback()

    def do_playback(self) -> None:
        """Playback saved video frames: to be called continuously"""
        if self._forward_one_frame():
            self.tk_scale.set(self._frame_idx + 1)
        else:
            self.stop_playback()

    def stop_playback(self) -> None:
        """Stop output playback"""
        self.is_output_playback = False
        self.set_viewer_state_to_stop()
        self._set_header_stop()

    def set_viewer_state_to_play(self) -> None:
        """Set self state to play for either 1) pipeline execution or 2) playback"""
        self.state = "play"
        self.tk_btn_play["text"] = STOP_BUTTON_TEXT

    def set_viewer_state_to_stop(self) -> None:
        """Set self state to stop"""
        self.state = "stop"
        self.tk_btn_play["text"] = PLAY_BUTTON_TEXT
