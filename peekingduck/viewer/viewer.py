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

from typing import List, Tuple
from functools import partial
from pathlib import Path
import logging
import time

# imports to run PeekingDuck
import copy
import cv2
from io import StringIO
import numpy as np
import os
import traceback
from peekingduck.declarative_loader import DeclarativeLoader
from peekingduck.pipeline.pipeline import Pipeline

# import for PeekingDuck Viewer
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import ttk
from peekingduck.viewer.canvas_view import CanvasView

LOGO = "peekingduck/viewer/AISG_Logo_1536x290.png"
WIN_WIDTH = 1280
WIN_HEIGHT = 800
FPS_60: int = int(1000 / 60)  # milliseconds per iteration
# FPS_60 = 500
BUTTON_DELAY: int = 250  # milliseconds (0.25 of a second)
BUTTON_REPEAT: int = int(1000 / 60)  # milliseconds (60 fps)
ZOOMS = [0.5, 0.75, 1.0, 1.25, 1.50, 2.00, 2.50, 3.00]  # > 3x is slow!
# Test unicode glyphs for zoom factors
# ZOOM_TEXT = ["\u00BD", "\u00BE", "1.0", "1\u00BC", "1\u00BD", "2.0"]
ZOOM_TEXT = ["0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x", "2.5x", "3x"]
#
# Tk Mapping Stuff
#
KEY_STATE_MAP = {
    1: "shift",
    2: "capslock",
    4: "ctrl",
    8: "meta",
    16: "alt",
}
MOUSE_BTN_MAP = {
    1: "left",
    2: "right",
    3: "middle",
}


def parse_streams(strio: StringIO) -> str:
    """Helper method to parse I/O streams.
    Used to capture errors/exceptions from PeekingDuck.

    Args:
        strio (StringIO): the I/O stream to parse

    Returns:
        str: parsed stream
    """
    msg = strio.getvalue()
    msg = os.linesep.join([s for s in msg.splitlines() if s])
    return msg


class Viewer:
    def __init__(
        self,
        pipeline_path: Path = None,
        config_updates_cli: str = None,
        custom_nodes_parent_subdir: str = None,
        num_iter: int = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._config_updates_cli = config_updates_cli
        self._custom_nodes_parent_path = custom_nodes_parent_subdir
        self._num_iter = num_iter
        # for PeekingDuck pipeline run/playback
        self.frames: List[np.ndarray] = None
        self.frame_idx: int = -1
        self.zoom_idx: int = 2
        self._output_playback: bool = False
        self._pipeline_path = pipeline_path
        self._pipeline_running: bool = False
        self._state: str = "play"  # activate auto play
        self._image_container = None

    def run(self):
        self.logger.info(f"run {self._pipeline_path}")
        self.logger.info(f"cwd={Path.cwd()}")
        logo_path = Path(LOGO)
        self.logger.info(f"{logo_path} exists={logo_path.exists()}")

        self._create_window()
        self._create_header()
        self._create_canvas()
        self._create_footer()
        self._create_progress_slider()

        # bind event handlers
        self.root.bind("<Key>", self.on_keypress)
        # self.root.bind("<Configure>", self.on_resize)
        # self.root.bind("<Button>", self.on_mousedown)
        # self.root.bind("<ButtonRelease>", self.on_mouseup)

        # self._timer = tk.Label(self.root)
        self.tick()

        self.root.mainloop()

    def _create_canvas(self) -> None:
        """Create the main image for viewing PeekingDuck output
        using Label widget"""
        canvas_frm = ttk.Frame(master=self.root)
        canvas_frm.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Label(canvas_frm)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.tk_pkd_image = canvas
        self.canvas_frm = canvas_frm

    def _create_canvas_old(self) -> None:
        """Create the main image for viewing PeekingDuck output
        using Canvas widget"""
        canvas_frm = ttk.Frame(master=self.root)
        canvas_frm.pack(fill=tk.BOTH, expand=True)
        # canvas = CanvasView(canvas_frm, bg="blue")
        canvas = CanvasView(canvas_frm)
        canvas.pack(fill=tk.BOTH, expand=True)
        self.tk_pkd_image = canvas
        self.canvas_frm = canvas_frm

    def _create_window(self) -> None:
        """Create the PeekingDuck viewer window"""
        root = tk.Tk()
        root.title = "PeekingDuck Viewer"
        root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        root.update()  # force update without mainloop()
        root.minsize(root.winfo_width(), root.winfo_height())
        # save it
        self.root = root

    def _create_header(self) -> None:
        """Create header with logo and info text"""
        header_frm = ttk.Frame(master=self.root)
        header_frm.pack(side=tk.TOP, fill=tk.X)
        # header contents
        self.img_logo = self._load_image(resize_pct=0.15)  # prevent garbage collection
        logo = tk.Label(header_frm, image=self.img_logo)
        logo.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            dummy = tk.Label(header_frm, text="")
            dummy.grid(row=0, column=i + 2, sticky="nsew")
        lbl = tk.Label(header_frm, text="PeekingDuck Viewer Header")
        lbl.grid(row=0, column=1, columnspan=3, sticky="nsew")
        self.tk_lbl_header = lbl
        # trick to do background processing with this widget
        lbl_timer = tk.Label(header_frm, text="timer")
        lbl_timer.grid(row=0, column=4, sticky="nsew")
        self.tk_lbl_timer = lbl_timer
        # configure expansion and uniform column sizes
        num_col, _ = header_frm.grid_size()
        for i in range(num_col):
            header_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        # save it
        self.header_frm = header_frm

    def _create_progress_slider(self) -> None:
        """Create progress bar/slider, frame count, zoom"""
        progress_frm = ttk.Frame(master=self.root)
        progress_frm.pack(side=tk.BOTTOM, fill=tk.X)
        lbl = tk.Label(progress_frm, text="")  # spacer
        lbl.grid(row=0, column=0, columnspan=3, sticky="nsew")
        # slider
        self.tk_scale = tk.Scale(
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
        # self.tk_frame_num = tk.Label(progress_frm, text="0")
        # self.tk_frame_num.grid(row=0, column=9, sticky="nsew")
        glyph = ZOOM_TEXT[self.zoom_idx]
        self.tk_lbl_zoom = tk.Label(progress_frm, text=f"{glyph}")
        self.tk_lbl_zoom.grid(row=0, column=9, sticky="nsew")
        lbl = tk.Label(progress_frm, text="")  # spacer
        lbl.grid(row=0, column=10, columnspan=2, sticky="nsew")
        # configure expansion and uniform column sizes
        num_col, _ = progress_frm.grid_size()
        for i in range(num_col):
            progress_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        # save it
        self.progress_frm = progress_frm

    def _create_footer(self) -> None:
        """Create footer of control buttons"""
        btn_frm = ttk.Frame(master=self.root)
        btn_frm.pack(side=tk.BOTTOM, fill=tk.X)
        # footer contents
        self.tk_btn_play = tk.Button(
            btn_frm, text="Play", command=self.btn_play_stop_press
        )
        btn_list = [
            tk.Label(btn_frm, text=""),  # spacer
            tk.Label(btn_frm, text=""),
            tk.Label(btn_frm, text=""),
            # tk.Button(btn_frm, text="Play", command=self.btn_play_stop_press),
            self.tk_btn_play,
            tk.Button(btn_frm, text="|<<", command=self.btn_first_frame_press),
            tk.Button(
                btn_frm,
                text="<<",
                command=self.btn_backward_press,
                repeatdelay=BUTTON_DELAY,
                repeatinterval=BUTTON_REPEAT,
            ),
            tk.Button(
                btn_frm,
                text=">>",
                command=self.btn_forward_press,
                repeatdelay=BUTTON_DELAY,
                repeatinterval=BUTTON_REPEAT,
            ),
            tk.Button(btn_frm, text=">>|", command=self.btn_last_frame_press),
            tk.Button(btn_frm, text="-", command=self.btn_zoom_out_press),
            tk.Button(btn_frm, text="+", command=self.btn_zoom_in_press),
            # tk.Button(btn_frm, text="Quit", command=self.btn_quit_press),
            tk.Label(btn_frm, text=""),
            tk.Label(btn_frm, text=""),
        ]
        for i, btn in enumerate(btn_list):
            btn.configure(height=2)  # NB: height is in text units!
            btn.grid(row=0, column=i, sticky="nsew")
            # btn.grid(row=0, column=i, ipady=10, ipadx=10, pady=10, padx=10)

        # configure expansion and uniform column sizes
        num_col, _ = btn_frm.grid_size()
        for i in range(num_col):
            btn_frm.grid_columnconfigure(i, weight=1, uniform="tag")
        # save it
        self.footer_frm = btn_frm

    def _load_image(self, resize_pct: float = 0.25) -> ImageTk.PhotoImage:
        """Load and resize an image

        Args:
            resize_pct (float, optional): percentage to resize. Defaults to 0.25.

        Returns:
            ImageTk.PhotoImage: the loaded image
        """
        img = Image.open(LOGO)
        w = int(resize_pct * img.size[0])
        h = int(resize_pct * img.size[1])
        resized_img = img.resize((w, h))
        # self.logger.info(f"img size={img.size}")
        the_img = ImageTk.PhotoImage(resized_img)
        return the_img

    #
    # Tk Event Handlers
    #
    def btn_play_stop_press(self):
        self.logger.info("btn_play_stop_press")
        if self._pipeline_running:
            self._stop_running_pipeline()
        elif self._output_playback:
            self._stop_playback()
        else:
            self.logger.info(f"self._state={self._state}")
            self._do_playback()
        self.logger.info(f"btn_play_stop_press end: self._state={self._state}")

    def btn_first_frame_press(self):
        if self._pipeline_running or self._output_playback or self.frames is None:
            return
        self.logger.debug("btn_first_frame_press")
        self.frame_idx = 0
        self._sync_frame_to_slider()
        self._show_frame()

    def btn_last_frame_press(self):
        if self._pipeline_running or self._output_playback or self.frames is None:
            return
        self.logger.debug("btn_last_frame_press")
        self.frame_idx = len(self.frames) - 1
        self._sync_frame_to_slider()
        self._show_frame()

    def btn_forward_press(self):
        if self._pipeline_running or self._output_playback or self.frames is None:
            return
        self._forward_one_frame()

    def btn_backward_press(self):
        if self._pipeline_running or self._output_playback or self.frames is None:
            return
        self._backward_one_frame()

    def btn_zoom_in_press(self):
        """Zoom in: make image larger"""
        self.logger.info("btn_zoom_in_press")
        if self.zoom_idx + 1 < len(ZOOMS):
            self.zoom_idx += 1
            self._update_zoom_text()

    def btn_zoom_out_press(self):
        """Zoom out: make image smaller"""
        self.logger.info("btn_zoom_out_press")
        if self.zoom_idx > 0:
            self.zoom_idx -= 1
            self._update_zoom_text()

    def btn_quit_press(self):
        self.logger.debug("btn_quit_press")
        self.root.destroy()

    def on_keypress(self, event):
        self.logger.info(
            f"keypressed: char={event.char}, keysym={event.keysym}, state={event.state}"
        )
        mod = KEY_STATE_MAP[event.state] if event.state in KEY_STATE_MAP else "unknown"
        if event.char or mod == "ctrl":  # make sure we have a key
            key = event.char if event.char else event.keysym
            self.logger.info(f"key = {mod}-{key}")
            # handle supported keyboard shortcuts here
            if mod == "ctrl" and key == "z":
                # reset zoom
                self.zoom_idx = 2
                self._update_zoom_text()

    # def on_mousedown(self, event):
    #     mouse_btn = MOUSE_BTN_MAP[event.num]
    #     self.logger.info(f"mousedown event={event} button={mouse_btn}")

    # def on_mouseup(self, event):
    #     mouse_btn = MOUSE_BTN_MAP[event.num]
    #     self.logger.info(f"mouseup event={event} button={mouse_btn}")

    # def on_resize(self, event):
    #     # self.logger.info(f"resize: event={event}")
    #     # self.logger.info(f"canvas: size={self.canvas.width}x{self.canvas.height}")
    #     if self._image_container:
    #         coords = self.canvas.coords(self._image_container)
    #         x1, y1 = coords
    #         x2 = self.canvas.width // 2
    #         y2 = self.canvas.height // 2
    #         dx = x2 - x1
    #         dy = y2 - y1
    #         if dx or dy:
    #             self.logger.info(
    #                 f"image: pos={coords}, x2={x2}, y2={y2}, dx={dx}, dy={dy}"
    #             )
    #             self.canvas.move(self._image_container, dx, dy)

    #
    # Background "Event Loop"
    #
    def tick(self):
        """Main background processing entry point"""
        the_time = time.strftime("%H:%M:%S")
        self.tk_lbl_timer.config(text=the_time)
        # self.logger.info(f"tick {the_time}, self._state={self._state}")

        if self._state == "play":
            if self._output_playback:
                self._do_playback()
            elif not self._pipeline_running:
                self._run_pipeline_start()
            elif self.pipeline.terminate:
                self._run_pipeline_end()
            else:
                self._run_pipeline_one_iteration()

        self.root.update()  # wake up GUI
        self.tk_lbl_timer.after(FPS_60, self.tick)

    def _backward_one_frame(self) -> bool:
        """Internal method to move back one frame, can be called repeatedly"""
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self._sync_frame_to_slider()
            self._show_frame()
            return True
        return False

    def _forward_one_frame(self) -> bool:
        """Internal method to move forward one frame, can be called repeatedly"""
        if self.frame_idx + 1 < len(self.frames):
            self.frame_idx += 1
            self._sync_frame_to_slider()
            self._show_frame()
            return True
        return False

    def _show_frame(self) -> None:
        if self.frames:
            frame = self.frames[self.frame_idx]
            frame = self._apply_zoom(frame)  # note: can speed up zoom?
            # self.logger.info(f"show_frame {self.frame_idx} size={frame.shape}")
            img_arr = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img_arr)
            # self.logger.info(f"img_tk: {img_tk.width()}x{img_tk.height()}")

            self._img_tk = img_tk  # save to avoid GC

            # using label
            self.tk_pkd_image.config(image=img_tk)
            # self.canvas.update()

            # # using actual canvas widget
            # ox = self.canvas.width // 2
            # oy = self.canvas.height // 2
            # # self.logger.info(
            # #     f"canvas: {self.canvas.width}x{self.canvas.height}, origin=({ox}, {oy})"
            # # )
            # if self._image_container:
            #     self.canvas.itemconfig(self._image_container, image=img_tk)
            # else:
            #     # self._image_container = self.canvas.create_image(
            #     #     ox, oy, anchor="center", image=img_tk
            #     # )
            #     self._image_container = self.canvas.create_image(ox, oy, image=img_tk)
            # self.canvas.update()  # need this to show image

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Zoom output image in real-time

        Args:
            frame (np.ndarray): image frame data to be zoomed

        Returns:
            np.ndarray: the zoomed image
        """
        # logger.debug(f"zoom_idx={self.zoom_idx}")
        if self.zoom_idx != 2:
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

    #
    # Header update methods
    #
    def _set_header_playing(self) -> None:
        self.tk_lbl_header["text"] = f"Playing {self._pipeline_path}"
        self.tk_lbl_header.config(fg="green")

    def _set_header_running(self) -> None:
        self.tk_lbl_header["text"] = f"Running {self._pipeline_path}"
        self.tk_lbl_header.config(fg="red")

    def _set_header_stop(self) -> None:
        self.tk_lbl_header["text"] = f"{self._pipeline_path}"
        self.tk_lbl_header.config(fg="white")

    def _sync_frame_to_slider(self) -> None:
        self.tk_scale.set(self.frame_idx + 1)

    def _sync_slider_to_frame(self, val) -> None:
        # self.logger.info(f"sync slider: {val}")
        # self.tk_frame_num["text"] = val
        self.frame_idx = int(val) - 1
        self._show_frame()

    def _enable_progress(self) -> None:
        self.logger.info("enable progress")
        self.tk_scale.grid_remove()  # hide slider
        self.tk_progress.grid()  # show progress bar

    def _enable_slider(self) -> None:
        self.logger.info("enable slider")
        self.tk_progress.grid_remove()  # hide progress bar
        self.tk_scale.grid()  # show slider
        self.tk_scale.configure(to=len(self.frames))
        # self.tk_frame_num["text"] = self.tk_scale.get()
        self._sync_slider_to_frame(self.tk_scale.get())

    def _run_pipeline_end(self) -> None:
        """Called when pipeline execution is completed.
        To perform clean-up/housekeeping tasks to ensure system consistency"""
        # self.logger.info("run pipeline end")
        for node in self.pipeline.nodes:
            if node.name.endswith("input.visual"):
                node.release_resources()  # clean up nodes with threads
        self._pipeline_running = False
        # self._toggle_btn_play_stop(state="play")
        # self.output_layout.install_slider()
        self._enable_slider()
        self._state = "stop"
        self.tk_btn_play["text"] = "Play"
        self._set_header_stop()

    def _run_pipeline_one_iteration(self) -> None:
        # self.logger.info("run pipeline one iteration")
        self._pipeline_running = True
        for node in self.pipeline.nodes:
            if self.pipeline.data.get("pipeline_end", False):
                self.pipeline.terminate = True
                if "pipeline_end" not in node.inputs:
                    continue
            if "all" in node.inputs:
                inputs = copy.deepcopy(self.pipeline.data)
            else:
                inputs = {
                    key: self.pipeline.data[key]
                    for key in node.inputs
                    if key in self.pipeline.data
                }
            if hasattr(node, "optional_inputs"):
                for key in node.optional_inputs:
                    # The nodes will not receive inputs with the optional
                    # key if it's not found upstream
                    if key in self.pipeline.data:
                        inputs[key] = self.pipeline.data[key]
            if node.name.endswith("output.screen"):
                # intercept screen output to Kivy
                img = self.pipeline.data["img"]
                # (0,0) == opencv top-left == kivy bottom-left
                # frame = cv2.flip(img, 0)  # flip around x-axis
                # convert from BGR to RGB for Tkinter
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)  # save frame for playback
                self.frame_idx += 1
                self._show_frame()
            else:
                outputs = node.run(inputs)
                self.pipeline.data.update(outputs)
            # check for FPS on first iteration
            if self.frame_idx == 0 and node.name.endswith("input.visual"):
                num_frames = node.total_frame_count
                if num_frames > 0:
                    self.num_frames = num_frames
                    self.tk_progress["maximum"] = num_frames
                else:
                    self.num_frames = 0
                    # self.progress = None
                    self.tk_progress["mode"] = "indeterminate"
        # update progress bar after each iteration
        self.tk_progress["value"] = self.frame_idx
        # self.tk_frame_num["text"] = self.tk_progress["value"]

    def _run_pipeline_start(self) -> None:
        """Init PeekingDuck's pipeline"""
        # self.logger.info("run pipeline start")
        self.logger.info(f"pipeline path: {self._pipeline_path}")
        self.logger.info(f"custom_nodes: {self._custom_nodes_parent_path}")
        self.node_loader = DeclarativeLoader(
            self._pipeline_path, "None", self._custom_nodes_parent_path
        )
        self.pipeline: Pipeline = self.node_loader.get_pipeline()
        # self.logger.info(f"self.pipeline: {self.pipeline}")
        self._set_header_running()

        self.frames = []
        self.frame_idx = -1
        self.tk_btn_play["text"] = "Stop"
        self._pipeline_running = True

    def _do_playback(self) -> None:
        self._output_playback = True
        self._state = "play"
        self.tk_btn_play["text"] = "Stop"
        self._set_header_playing()
        if self._forward_one_frame():
            self.tk_scale.set(self.frame_idx + 1)
            self.logger.debug("forward one frame ok")
        else:
            self._stop_playback()

    def _stop_playback(self) -> None:
        """Stop output playback"""
        if hasattr(self, "forward_one_frame_held"):
            self.forward_one_frame_held.cancel()
        self._output_playback = False
        self._state = "stop"
        self.tk_btn_play["text"] = "Play"
        self._set_header_stop()

    def _stop_running_pipeline(self) -> None:
        """Signal pipeline execution to be stopped"""
        self.pipeline.terminate = True

    def _update_zoom_text(self) -> None:
        """Databinding for zoom -> image"""
        glyph = ZOOM_TEXT[self.zoom_idx]
        self.logger.info(f"Zoom: {glyph}")
        self.tk_lbl_zoom["text"] = f"{glyph}"
        self._show_frame()
