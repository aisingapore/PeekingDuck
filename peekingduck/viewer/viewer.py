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

from typing import List, Union
from pathlib import Path
import logging
import platform
import time
import tkinter as tk
from tkinter import ttk
import copy
import cv2
import numpy as np
from PIL import ImageTk, Image
from peekingduck.declarative_loader import DeclarativeLoader
from peekingduck.pipeline.pipeline import Pipeline
from peekingduck.viewer.viewer_utils import (
    load_image,
    get_keyboard_char,
    get_keyboard_modifier,
)

####################
# Globals
####################
BUTTON_DELAY: int = 250  # milliseconds (0.25 of a second)
BUTTON_REPEAT: int = int(1000 / 60)  # milliseconds (60 fps)
FPS_60: int = int(1000 / 60)  # milliseconds per iteration
LOGO = "peekingduck/viewer/PeekingDuckLogo.png"
WIN_HEIGHT = 600
WIN_WIDTH = 800
ZOOM_TEXT = ["0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x", "2.5x", "3x"]
ZOOM_DEFAULT_IDX = 2
ZOOMS = [0.5, 0.75, 1.0, 1.25, 1.50, 2.00, 2.50, 3.00]  # > 3x is slow!


class Viewer:  # pylint: disable=too-many-instance-attributes
    """Implement PeekingDuck Viewer class"""

    def __init__(
        self,
        pipeline_path: Path,
        config_updates_cli: str,
        custom_nodes_parent_subdir: str,
        num_iter: int = 0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._pipeline_path = pipeline_path
        self._config_updates_cli = config_updates_cli
        self._custom_nodes_parent_path = custom_nodes_parent_subdir
        self._num_iter = num_iter
        # for PeekingDuck pipeline run/playback
        self._frames: List[np.ndarray] = []
        self._frame_idx: int = -1
        self.zoom_idx: int = ZOOM_DEFAULT_IDX
        self._is_output_playback: bool = False
        self._is_pipeline_running: bool = False
        self._state: str = "play"  # activate auto play (cf. self.timer_function)
        self._bkgd_job: Union[None, str] = None
        # configure keyboard shortcuts map
        self._keyboard_shortcuts = {
            "z": self._zoom_reset,
            "+": self._zoom_in,
            "-": self._zoom_out,
        }

    def run(self) -> None:
        """Main method to setup Viewer and run Tk event loop"""
        self.logger.info(f"cwd={Path.cwd()}")
        self.logger.info(f"pipeline={self._pipeline_path}")
        logo_path = Path(LOGO)
        self.logger.debug(f"logo={logo_path}, exists={logo_path.exists()}")
        # create Tkinter window and frames
        self._create_window()
        self._create_header()
        self._create_canvas()
        self._create_footer()  # need to order last two frames correctly
        self._create_progress_slider()
        # bind event handlers
        self.root.bind("<Key>", self.on_keypress)
        if platform.system() == "Darwin":
            self.logger.info("binding macOS cmd-Q")
            self.root.createcommand("::tk::mac::Quit", self.on_exit)
        # activate internal timer function and start Tkinter event loop
        self._timer_function()
        self.root.mainloop()

    ####################
    #
    # Tk Main Window and Frames Creation
    #
    ####################
    def _create_window(self) -> None:
        """Create the PeekingDuck viewer window"""
        root = tk.Tk()
        root.wm_protocol("WM_DELETE_WINDOW", self.on_exit)
        root.title("PeekingDuck Viewer")
        root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        root.update()  # force update without mainloop() to get correct size
        root.minsize(root.winfo_width(), root.winfo_height())
        self.root = root  # save main window

    def _create_canvas(self) -> None:
        """Create the main image widget for viewing PeekingDuck output"""
        image_frm = ttk.Frame(master=self.root)
        image_frm.pack(fill=tk.BOTH, expand=True)
        output_image = tk.Label(image_frm)
        output_image.pack(fill=tk.BOTH, expand=True)
        self.tk_output_image = output_image
        self.image_frm = image_frm  # save image frame

    def _create_header(self) -> None:
        """Create header with logo and pipeline info text"""
        header_frm = ttk.Frame(master=self.root)
        header_frm.pack(side=tk.TOP, fill=tk.X)
        # header contents
        self._img_logo = load_image(LOGO, resize_pct=0.15)  # prevent python GC
        logo = tk.Label(header_frm, image=self._img_logo)
        logo.grid(row=0, column=0, sticky="w")
        for i in range(2):
            dummy = tk.Label(header_frm, text="")
            dummy.grid(row=0, column=i + 2, sticky="nsew")
        lbl = tk.Label(header_frm, text="PeekingDuck Viewer Header")
        lbl.grid(row=0, column=1, columnspan=3, sticky="nsew")
        self.tk_lbl_header = lbl
        # setup "timer" widget to do background processing later
        lbl_timer = tk.Label(header_frm, text="timer")
        lbl_timer.grid(row=0, column=4, sticky="e")
        self.tk_lbl_timer = lbl_timer
        # configure expansion and uniform column sizes
        num_col, _ = header_frm.grid_size()
        for i in range(num_col):
            header_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        self.header_frm = header_frm  # save header frame

    def _create_progress_slider(self) -> None:
        """Create frame for progress bar/slider, frame count, zoom.
        Need to align columns with footer below.
        Progress bar and slider overlaps on the same columns.
        """
        progress_frm = ttk.Frame(master=self.root)
        progress_frm.pack(side=tk.BOTTOM, fill=tk.X)
        lbl = tk.Label(progress_frm, text="")  # spacer
        lbl.grid(row=0, column=0, columnspan=2, sticky="nsew")
        # frame number
        self.tk_lbl_frame_num = tk.Label(progress_frm, text="0")
        self.tk_lbl_frame_num.grid(row=0, column=2, sticky="nsew")
        # slider
        self.tk_scale = ttk.Scale(
            progress_frm,
            orient=tk.HORIZONTAL,
            from_=1,
            to=100,
            command=self._sync_slider_to_frame,
        )
        self.tk_scale.grid(row=0, column=3, columnspan=6, sticky="nsew")
        self.tk_scale.grid_remove()  # hide it first
        # progress bar
        self.tk_progress = ttk.Progressbar(
            progress_frm,
            orient=tk.HORIZONTAL,
            length=100,
            mode="determinate",
            value=0,
            maximum=100,
        )
        self.tk_progress.grid(row=0, column=3, columnspan=6, sticky="nsew")
        # zoom
        glyph = ZOOM_TEXT[self.zoom_idx]
        self.tk_lbl_zoom = tk.Label(progress_frm, text=f"{glyph}")
        self.tk_lbl_zoom.grid(row=0, column=9, sticky="nsew")
        lbl = tk.Label(progress_frm, text="")  # spacer
        lbl.grid(row=0, column=10, columnspan=2, sticky="nsew")
        # configure expansion and uniform column sizes
        num_col, _ = progress_frm.grid_size()
        for i in range(num_col):
            progress_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        self.progress_frm = progress_frm  # save progress/slider frame

    def _create_footer(self) -> None:
        """Create footer of control buttons.
        Use blank tk.Label to guide spacing as Tkinter can't micromanage layout
        by specifying percentages/pixels.
        """
        btn_frm = ttk.Frame(master=self.root)
        btn_frm.pack(side=tk.BOTTOM, fill=tk.X)
        # footer contents
        self.tk_btn_play = ttk.Button(
            btn_frm, text="Play", command=self.btn_play_stop_press
        )  # store widget to modifying button text later
        btn_list: List[Union[ttk.Button, tk.Label]] = [
            tk.Label(btn_frm, text=""),  # spacer
            tk.Label(btn_frm, text=""),
            tk.Label(btn_frm, text=""),
            self.tk_btn_play,
            tk.Label(btn_frm, text=""),
            ttk.Button(btn_frm, text="-", command=self.btn_zoom_out_press),
            ttk.Button(btn_frm, text="+", command=self.btn_zoom_in_press),
            tk.Label(btn_frm, text=""),
            tk.Label(btn_frm, text=""),
            tk.Label(btn_frm, text=""),
        ]
        for i, btn in enumerate(btn_list):
            # btn.configure(height=2)  # NB: height in text units (N/A to ttk)
            btn.grid(row=0, column=i, sticky="nsew")
        # configure expansion and uniform column sizes
        num_col, _ = btn_frm.grid_size()
        for i in range(num_col):
            btn_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        self.footer_frm = btn_frm  # save footer frame

    ####################
    #
    # Tk Event Handlers
    #
    ####################
    def btn_play_stop_press(self) -> None:
        """Handle Play/Stop button"""
        self.logger.debug(f"btn_play_stop_press start: self._state={self._state}")
        if self._is_pipeline_running:
            self._stop_running_pipeline()
        elif self._is_output_playback:
            self._stop_playback()
        else:
            self._start_playback()
        self.logger.debug(f"btn_play_stop_press end: self._state={self._state}")

    def btn_first_frame_press(self) -> None:
        """Goto first frame"""
        if (
            self._is_pipeline_running
            or self._is_output_playback
            or self._frames is None
        ):
            return
        self.logger.debug("btn_first_frame_press")
        self._frame_idx = 0
        self._update_slider_and_show_frame()

    def btn_last_frame_press(self) -> None:
        """Goto last frame"""
        if (
            self._is_pipeline_running
            or self._is_output_playback
            or self._frames is None
        ):
            return
        self.logger.debug("btn_last_frame_press")
        self._frame_idx = len(self._frames) - 1
        self._update_slider_and_show_frame()

    def btn_forward_press(self) -> None:
        """Forward one frame"""
        if (
            self._is_pipeline_running
            or self._is_output_playback
            or self._frames is None
        ):
            return
        self._forward_one_frame()

    def btn_backward_press(self) -> None:
        """Back one frame"""
        if (
            self._is_pipeline_running
            or self._is_output_playback
            or self._frames is None
        ):
            return
        self._backward_one_frame()

    def btn_zoom_in_press(self) -> None:
        """Zoom in: make image larger"""
        self.logger.info("btn_zoom_in_press")
        self._zoom_in()

    def btn_zoom_out_press(self) -> None:
        """Zoom out: make image smaller"""
        self.logger.info("btn_zoom_out_press")
        self._zoom_out()

    def on_keypress(self, event: tk.Event) -> None:
        """Handle all keydown events.
        Default system shortcuts are automatically handled, e.g. CMD-Q quits on macOS

        Args:
            event (tk.Event): the key down event
        """
        self.logger.info(
            f"keypressed: char={event.char}, keysym={event.keysym}, state={event.state}"
        )
        key_state: int = int(event.state)
        mod = get_keyboard_modifier(key_state)
        key = get_keyboard_char(event.char, event.keysym)
        self.logger.info(f"mod={mod}, key={key}")
        # handle supported keyboard shortcuts here
        if mod.startswith("ctrl"):
            if key in self._keyboard_shortcuts:
                self._keyboard_shortcuts[key]()

    def on_exit(self) -> None:
        """Handle quit viewer event"""
        self.logger.info("quitting viewer")
        self._cancel_timer_function()
        self.root.destroy()

    #
    # Background "Event Loop"
    #
    def _timer_function(self) -> None:
        """Function to do background processing in Tkinter's way"""
        the_time = time.strftime("%H:%M:%S")
        self.tk_lbl_timer.config(text=the_time)
        # self.logger.debug(f"timer function: {the_time}, _state={self._state}")

        if self._state == "play":
            # Only two states: 1) playing back video or 2) executing pipeline
            if self._is_output_playback:
                self._do_playback()
            else:
                # Executing pipeline: check which execution state we are in
                if not self._is_pipeline_running:
                    self._run_pipeline_start()
                elif self._pipeline.terminate:
                    self._run_pipeline_end()
                else:
                    self._run_pipeline_one_iteration()

        self.root.update()  # wake up GUI
        self._bkgd_job = self.tk_lbl_timer.after(FPS_60, self._timer_function)

    def _cancel_timer_function(self) -> None:
        """Cancel the background timer function"""
        if self._bkgd_job:
            self.tk_lbl_timer.after_cancel(self._bkgd_job)
            self._bkgd_job = None

    ####################
    #
    # Internal Methods for Display Management
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
        # logger.debug(f"zoom_idx={self.zoom_idx}")
        if self.zoom_idx != ZOOM_DEFAULT_IDX:
            # zoom image
            zoom = ZOOMS[self.zoom_idx]
            # logger.debug(f"img.shape = {img.shape}, zoom = {zoom}")
            new_size = (
                int(frame.shape[0] * zoom),
                int(frame.shape[1] * zoom),
                frame.shape[2],
            )
            # logger.debug(f"zoom image to {new_size}")
            # note: opencv is faster than scikit-image!
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
        self.logger.info(f"Zoom: {glyph}")
        self.tk_lbl_zoom["text"] = f"{glyph}"
        self._show_frame()

    def _set_header_playing(self) -> None:
        """Change header text to playing..."""
        self.tk_lbl_header["text"] = f"Playing {self._pipeline_path}"
        self.tk_lbl_header.config(fg="green")

    def _set_header_running(self) -> None:
        """Change header text to running..."""
        self.tk_lbl_header["text"] = f"Running {self._pipeline_path}"
        self.tk_lbl_header.config(fg="red")

    def _set_header_stop(self) -> None:
        """Change header text to pipeline pathname"""
        self.tk_lbl_header["text"] = f"{self._pipeline_path}"
        self.tk_lbl_header.config(fg="white")

    def _update_slider_and_show_frame(self) -> None:
        """Update slider based on frame index and show new frame"""
        frame_num = self._frame_idx + 1
        self.tk_scale.set(frame_num)
        self.tk_lbl_frame_num["text"] = frame_num
        self._show_frame()

    def _sync_slider_to_frame(self, val: str) -> None:
        """Update frame index based on slider value change

        Args:
            val (str): slider value
        """
        self.logger.debug(f"sync slider to frame: {val} {type(val)}")
        self._frame_idx = round(float(val)) - 1
        self.tk_lbl_frame_num["text"] = self._frame_idx + 1
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
        self.tk_scale.configure(to=len(self._frames))
        self._sync_slider_to_frame(self.tk_scale.get())

    ####################
    #
    # Pipeline Execution Methods
    #
    ####################
    def _run_pipeline_end(self) -> None:
        """Called when pipeline execution is completed.
        To perform clean-up/housekeeping tasks to ensure system consistency"""
        self.logger.debug("run pipeline end")
        for node in self._pipeline.nodes:
            if node.name.endswith("input.visual"):
                node.release_resources()  # clean up nodes with threads
        self._is_pipeline_running = False
        self._enable_slider()
        self._set_viewer_state_to_stop()
        self._set_header_stop()

    def _run_pipeline_one_iteration(self) -> None:  # pylint: disable=too-many-branches
        self._is_pipeline_running = True
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
                # intercept screen output to Tkinter
                img = self._pipeline.data["img"]
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB for Tkinter
                self._frames.append(frame)  # save frame for playback
                self._frame_idx += 1
                self._show_frame()
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
            # check if need to stop after fixed number of iterations
            if self._num_iter and self._frame_idx + 1 >= self._num_iter:
                self.logger.info(f"Stopping pipeline after {self._num_iter} iterations")
                self._stop_running_pipeline()
        # update progress bar after each iteration
        self.tk_progress["value"] = self._frame_idx
        self.tk_lbl_frame_num["text"] = self._frame_idx + 1

    def _run_pipeline_start(self) -> None:
        """Init PeekingDuck's pipeline"""
        self.logger.debug("run pipeline start")
        self.logger.debug(f"pipeline path: {self._pipeline_path}")
        self.logger.debug(f"custom_nodes: {self._custom_nodes_parent_path}")
        self._node_loader = DeclarativeLoader(
            self._pipeline_path,
            self._config_updates_cli,
            self._custom_nodes_parent_path,
        )
        self._pipeline: Pipeline = self._node_loader.get_pipeline()
        self._set_header_running()
        self._set_viewer_state_to_play()
        self._is_pipeline_running = True

    def _stop_running_pipeline(self) -> None:
        """Signal pipeline execution to be stopped"""
        self._pipeline.terminate = True

    ####################
    #
    # Pipeline Playback Methods
    #
    ####################
    def _start_playback(self) -> None:
        """Start output playback process"""
        self._is_output_playback = True
        # auto-rewind if at last frame
        if self._frame_idx + 1 >= len(self._frames):
            self._frame_idx = 0
            self._update_slider_and_show_frame()
        self._set_viewer_state_to_play()
        self._set_header_playing()
        self._do_playback()

    def _do_playback(self) -> None:
        """Playback saved video frames: to be called continuously"""
        if self._forward_one_frame():
            self.tk_scale.set(self._frame_idx + 1)
        else:
            self._stop_playback()

    def _stop_playback(self) -> None:
        """Stop output playback"""
        self._is_output_playback = False
        self._set_viewer_state_to_stop()
        self._set_header_stop()

    def _set_viewer_state_to_play(self) -> None:
        """Set self state to play for either 1) pipeline execution or 2) playback"""
        self._state = "play"
        self.tk_btn_play["text"] = "Stop"

    def _set_viewer_state_to_stop(self) -> None:
        """Set self state to stop"""
        self._state = "stop"
        self.tk_btn_play["text"] = "Play"
